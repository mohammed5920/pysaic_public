import os
from PIL import Image

for file in os.listdir():
    if file.endswith("ico"):
        img = Image.open(file).convert("RGBA")
        alpha = img.getchannel("A")
        bbox = alpha.getbbox()
        img = img.crop(bbox)
        img.save(f"{file[:-3]}png")
        os.remove(file)