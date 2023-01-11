from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
app = FastAPI()

@app.get("/isalive")
def isalive():
    return {"message" : True}

@app.post("/impaint")
async def impaint(image : UploadFile, mask : UploadFile, prompt : str):
    image_bytes = await image.read()
    image = Image.open(io.BytesIO(image_bytes))
    mask_bytes = await mask.read()
    mask = Image.open(io.BytesIO(mask_bytes))

    return {"Image/Mask" : f"{image.filename}/{mask.filename}"}

@app.post("/txt2img")
async def txt2img(txt : str):
    return {"message": "Yo!"}