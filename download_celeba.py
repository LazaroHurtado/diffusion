import io
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

HF_BASE = "https://huggingface.co/datasets/mattymchen/celeba-hq/resolve/main/data"
OUTPUT_DIR = Path("celeba_hq")

SHARDS = [
    "train-00000-of-00006-bae07ad6d4d89a77.parquet",
    "train-00001-of-00006-77346f9096557aa8.parquet",
    "train-00002-of-00006-e8a1953384ecd6a7.parquet",
    "train-00003-of-00006-fced104e22ee3f82.parquet",
    "train-00004-of-00006-80e5c44b7b2119cf.parquet",
    "train-00005-of-00006-2ad819c3650dbd46.parquet",
    "validation-00000-of-00001-a47eeed86041f01b.parquet",
]


def download_file(url: str, dest: Path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        total = int(r.headers.get("content-length", 0))
        bar = tqdm(total=total, unit="B", unit_scale=True, desc=dest.name, leave=False)
        while chunk := r.read(1 << 20):
            f.write(chunk)
            bar.update(len(chunk))
        bar.close()


def extract_images(parquet_path: Path, out_dir: Path, start_idx: int) -> int:
    table = pq.read_table(parquet_path)
    images_col = table.column("image")
    for i, entry in enumerate(images_col):
        raw = entry.as_py()
        img_bytes = raw["bytes"] if isinstance(raw, dict) else raw
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.save(out_dir / f"{start_idx + i:06d}.jpg")
    return len(images_col)


def main():
    cache_dir = OUTPUT_DIR / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    idx = 0
    for shard in SHARDS:
        cached = cache_dir / shard
        if not cached.exists():
            print(f"Downloading {shard} ...")
            download_file(f"{HF_BASE}/{shard}", cached)
        print(f"Extracting {shard} -> {OUTPUT_DIR} (starting at {idx:06d})")
        idx += extract_images(cached, OUTPUT_DIR, idx)
    print(f"{idx} images saved to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

