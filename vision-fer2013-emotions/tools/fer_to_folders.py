import csv
from pathlib import Path
from PIL import Image

LABELS = {"0":"angry","1":"disgust","2":"fear","3":"happy","4":"sad","5":"surprise","6":"neutral"}

def save_img(pixels_str, out_path: Path):
    vals = list(map(int, pixels_str.split()))
    img = Image.new("L", (48,48))
    img.putdata(vals)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)

def convert(csv_path: str, out_root: str):
    out_root = Path(out_root)
    counts = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            usage = row["Usage"].lower()
            split = "train" if "train" in usage else ("val" if "public" in usage else "test")
            label = LABELS[row["emotion"]]
            idx = counts.get((split,label), 0)
            out = out_root / split / label / f"{label}_{idx:06d}.png"
            save_img(row["pixels"], out)
            counts[(split,label)] = idx+1
    print("Wrote images to", out_root)

if __name__ == "__main__":
    import sys
    convert(sys.argv[1], sys.argv[2])
