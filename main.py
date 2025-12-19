from fastapi import FastAPI, UploadFile, File
import shutil
import uuid
from service import diagnose

app = FastAPI()

@app.post("/diagnose")
async def diagnose_plant(image: UploadFile = File(...)):
    filename = f"temp_{uuid.uuid4()}.jpg"
    with open(filename, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    result = diagnose(filename)
    return result
