from fastapi import FastAPI

from backend.app.routers import jobs
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5678",
        "http://127.0.0.1:5678",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



app.include_router(jobs.router, prefix="/job", tags=["launch-job"],)

@app.get("/")
def root():
    return {"status": "P2MC API OK"}


#uvicorn app.main:app --host 0.0.0.0 --port $PORT
