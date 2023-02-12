FROM continuumio/miniconda3

WORKDIR /SD2_Server
# Download model checkpoint
RUN wget -P ./models/ https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/resolve/main/512-inpainting-ema.ckpt
# For OpenCV install libGL
RUN apt-get update && apt-get install -y libgl1
# Update base env with required packages
COPY ./environment.yaml /SD2_Server/environment.yaml
RUN conda env update -f environment.yaml --prune
# Copy code repo
COPY . .
# Start SD2 server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "80"]
