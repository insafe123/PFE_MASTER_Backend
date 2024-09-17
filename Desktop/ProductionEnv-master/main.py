from fastapi import FastAPI
import os
from ModelRouter import model_route
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000",
]


#stage = os.envirofn.get('STAGE', 'dev')
app = FastAPI()
app.include_router(router=model_route, prefix="/model")


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello Doctor"}


