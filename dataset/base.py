from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class BaseDataset(Dataset):
    img_shape = None
    label_column = None

    def __init__(self, train=True):
        self.train = train
        self.transform = self._transform()
        self.ds = self._load()

    def _transform(self):
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def _load(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]

        img = item["image"]
        if self.transform:
            img = self.transform(img)

        label = -1
        if self.label_column:
            label = item[self.label_column]

        return img, label

    def to_dataloader(
        self,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
    ):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers and num_workers > 0,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )
