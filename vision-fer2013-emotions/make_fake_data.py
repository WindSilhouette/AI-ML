# make_fake_data.py
from PIL import Image, ImageDraw
import numpy as np, os

def make_img(seed, label):
    rng = np.random.default_rng(seed)
    img = Image.fromarray((rng.normal(128, 40, size=(64,64)).clip(0,255)).astype(np.uint8))
    d = ImageDraw.Draw(img)
    if label=="happy":
        d.arc((16,24,48,56), start=200, end=340, fill=255, width=2)
    else:
        d.arc((16,34,48,66), start=20, end=160, fill=255, width=2)
    d.ellipse((20,18,26,24), fill=255)
    d.ellipse((38,18,44,24), fill=255)
    return img

os.makedirs("data/sample/happy", exist_ok=True)
os.makedirs("data/sample/sad", exist_ok=True)

for i in range(12):
    make_img(i,"happy").save(f"data/sample/happy/happy_{i}.png")
    make_img(100+i,"sad").save(f"data/sample/sad/sad_{i}.png")

print("Wrote synthetic images.")

