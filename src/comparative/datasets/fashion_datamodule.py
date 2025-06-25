# src/comparative/datasets/fashion_datamodule.py

import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pytorch_lightning as pl
from transformers import AutoTokenizer
import torchvision.transforms as T

class FashionAIDataset(Dataset):
    def __init__(self, csv_path, tokenizer_name, max_len, img_size=224):
        df = pd.read_csv(csv_path)
        self.img_paths = df['image_path'].tolist()
        self.texts = df['description'].fillna("").astype(str).tolist()
        self.labels = df['label'].astype(str).tolist()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])
        self.label2idx = {lbl: i for i, lbl in enumerate(sorted(set(self.labels)))}
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.img_paths[idx]).convert('RGB')
        img = self.transform(img)
        # Tokenize text
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['image'] = img
        item['labels'] = self.label2idx[self.labels[idx]]
        return item

class FashionAIDatamodule(pl.LightningDataModule):
    def __init__(self, data_dir, tokenizer_name="distilbert-base-uncased", batch_size=32, max_len=128, img_size=224, num_workers=12):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.max_len = max_len
        self.img_size = img_size
        self.num_workers = num_workers
    def setup(self, stage=None):
        self.train_ds = FashionAIDataset(self.data_dir / "train.csv", self.tokenizer_name, self.max_len, self.img_size)
        self.val_ds = FashionAIDataset(self.data_dir / "val.csv", self.tokenizer_name, self.max_len, self.img_size)
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
