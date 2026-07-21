import os
import time

import requests

from ..config import GROBID_URL


REQUEST_TIMEOUT_SECONDS = 600
GROBID_STARTUP_TIMEOUT_SECONDS = int(
    os.getenv("GROBID_STARTUP_TIMEOUT_SECONDS", str(REQUEST_TIMEOUT_SECONDS))
)
GROBID_HEALTHCHECK_TIMEOUT_SECONDS = int(
    os.getenv("GROBID_HEALTHCHECK_TIMEOUT_SECONDS", "5")
)
GROBID_RETRY_DELAY_SECONDS = int(
    os.getenv("GROBID_RETRY_DELAY_SECONDS", "5")
)


class SciPdfParser:
    def __init__(self, grobid_url=GROBID_URL):
        """
        Inicializa el parser y verifica la conexion con el servidor GROBID
        al instanciar la clase.
        """
        print("SciPdfParser: initializing and checking GROBID connection...")
        self.grobid_url = grobid_url
        self._check_connection()

    def _check_connection(self):
        """Verifica si el servidor GROBID esta activo."""
        deadline = time.monotonic() + GROBID_STARTUP_TIMEOUT_SECONDS
        last_error: Exception | None = None

        while True:
            try:
                remaining_seconds = max(1.0, deadline - time.monotonic())
                response = requests.get(
                    f"{self.grobid_url}/api/isalive",
                    timeout=min(
                        GROBID_HEALTHCHECK_TIMEOUT_SECONDS,
                        remaining_seconds,
                    ),
                )
                if response.status_code == 200:
                    print(f"Connected to GROBID at {self.grobid_url}")
                    return

                last_error = RuntimeError(
                    f"GROBID at {self.grobid_url} returned status "
                    f"{response.status_code}"
                )

            except requests.exceptions.RequestException as exc:
                last_error = exc

            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise RuntimeError(
                    f"Could not connect to GROBID at {self.grobid_url}"
                ) from last_error

            sleep_seconds = min(
                GROBID_RETRY_DELAY_SECONDS,
                remaining_seconds,
            )
            print(
                "GROBID not ready at "
                f"{self.grobid_url}; retrying in "
                f"{sleep_seconds:.0f}s..."
            )
            time.sleep(sleep_seconds)

    def process(self, pdf_path, xml_output_path):
        """
        Procesa un unico PDF usando el backend de SciPDF (GROBID)
        y guarda el XML resultante.
        """
        if not pdf_path or not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if os.path.exists(xml_output_path):
            print(
                "XML already exists. Skipping extraction for: "
                f"{os.path.basename(xml_output_path)}"
            )
            return xml_output_path

        endpoint = f"{self.grobid_url}/api/processFulltextDocument"

        try:
            with open(pdf_path, "rb") as file:
                files = {"input": file}
                data = {
                    "generateTeiIds": "1",
                    "consolidateHeader": "1",
                    "consolidateConclusion": "1",
                }

                response = requests.post(
                    endpoint,
                    files=files,
                    data=data,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )

            if response.status_code != 200:
                raise RuntimeError(
                    f"GROBID failed to process {pdf_path}: "
                    f"HTTP {response.status_code}"
                )

            os.makedirs(os.path.dirname(xml_output_path), exist_ok=True)

            with open(xml_output_path, "w", encoding="utf-8") as xml_file:
                xml_file.write(response.text)

            return xml_output_path

        except Exception as exc:
            raise RuntimeError(
                f"Critical error processing {pdf_path} with GROBID: {exc}"
            ) from exc
