import base64
from io import BytesIO

from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image

from .base import BaseDataset


def _bytes_to_image(bytes_data):
    payload = base64.b64decode(bytes_data)
    return Image.open(BytesIO(payload)).convert("RGB")


class CelebSmallDataset(BaseDataset):
    img_shape = (3, 64, 64)

    def _load(self):
        return (
            load_dataset("BeibeiLim/celeba_hq_64", split="train")
            .cast_column("image", HFImage(decode=False))
            .with_transform(
                lambda x: {
                    "image": [_bytes_to_image(item["path"]) for item in x["image"]]
                }
            )
        )
