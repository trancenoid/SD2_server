from fastapi import FastAPI, UploadFile, Form, Response
from PIL import Image
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import io
from inpainting import *


class Txt2ImgModel(BaseModel):
    txt : str

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sampler = initialize_model('v2-inpainting-inference.yaml', 'models/512-inpainting-ema.ckpt')

@app.get("/isalive")
def isalive():
    return {"message" : True}

@app.post("/inpaint/")
async def inpaint(image : UploadFile, mask : UploadFile , prompt : str = Form()):
    print(image.filename,mask.filename,prompt)

    image_bytes = await image.read()
    image_ = Image.open(io.BytesIO(image_bytes))
    image_.save(image.filename)

    mask_bytes = await mask.read()
    mask_ = Image.open(io.BytesIO(mask_bytes))
    mask_.save(mask.filename)

    result = get_inpainted_image(image_,mask_,prompt,sampler)
    result[0].save("inpainted.png")

    png_bytes = io.BytesIO()
    result[0].save(png_bytes, format='PNG')

    return Response(content=png_bytes.getvalue(), status_code=200, media_type="image/png")

@app.post("/txt2img/")
async def txt2img(body : Txt2ImgModel):
    return {"message": f"Yo! you entered {body.txt}"}