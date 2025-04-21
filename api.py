from typing import Union
from fastapi import FastAPI

app = FastAPI()

@app.post("/verify")
async def verify():
    return {":"}


@app.post("/spoof-detect")
async def spoofDetect():
    return {":"}

@app.post("/uploadAudio")
async def register_spk():
    return {"Hello": "World"}