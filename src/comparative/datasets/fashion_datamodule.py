# src/comparative/datasets/fashion_datamodule.py

import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer

class FashionAIDataset(Dataset):
    def __init__(self, csv_path, image_emb_path, tokenizer_name, max_len, label2idx):
        df = pd.read_csv(csv_path)
        self.texts = df['description'].fillna("").astype(str).tolist()
        self.labels = df['label'].astype(str).tolist()
        self.img_embs = np.load(image_emb_path)
        assert len(self.texts) == len(self.img_embs), f"Image embedding length ({len(self.img_embs)}) does not match data ({len(self.texts)})!"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.label2idx = label2idx
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['image_emb'] = torch.tensor(self.img_embs[idx]).float()
        item['label'] = self.label2idx[self.labels[idx]]
        return item

class FashionAIDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        tokenizer_name="distilbert-base-uncased",
        batch_size=32,
        max_len=128,
        img_size=224,  # not used but kept for hydra compatibility!
        num_workers=12
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.max_len = max_len
        self.img_size = img_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Build label2idx on *union* of all splits (robust even if class missing in val/test)
        all_labels = set()
        for fname in ["train.csv", "val.csv", "test.csv"]:
            f = self.data_dir / fname
            if f.exists():
                all_labels.update(pd.read_csv(f)['label'].astype(str).unique())
        label2idx = {lbl: i for i, lbl in enumerate(sorted(all_labels))}
        self.label2idx = label2idx

        self.train_ds = FashionAIDataset(
            self.data_dir / "train.csv",
            self.data_dir / "train_image_emb.npy",
            self.tokenizer_name,
            self.max_len,
            label2idx
        )
        self.val_ds = FashionAIDataset(
            self.data_dir / "val.csv",
            self.data_dir / "val_image_emb.npy",
            self.tokenizer_name,
            self.max_len,
            label2idx
        )
        # Optional: test set (if exists)
        test_csv = self.data_dir / "test.csv"
        test_emb = self.data_dir / "test_image_emb.npy"
        if test_csv.exists() and test_emb.exists():
            self.test_ds = FashionAIDataset(test_csv, test_emb, self.tokenizer_name, self.max_len, label2idx)
        else:
            self.test_ds = None
        print(f"train: {len(self.train_ds)}, image_emb: {self.train_ds.img_embs.shape}")
        print(f"val: {len(self.val_ds)}, image_emb: {self.val_ds.img_embs.shape}")


    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
    def test_dataloader(self):
        if self.test_ds is not None:
            return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        return None
