# src/comparative/datasets/amazon_datamodule.py

import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer

class AmazonReviewDataset(Dataset):
    def __init__(self, csv_path, tokenizer_name, max_len, n_samples=None):
        df = pd.read_csv(csv_path)
        if n_samples is not None:
            df = df.iloc[:n_samples].reset_index(drop=True)
        self.texts = df["full_text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['label'] = label - 1  # 0-indexed labels
        return item

class AmazonDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        tokenizer_name="distilbert-base-uncased",
        batch_size=32,
        max_len=128,
        num_workers=12,
        train_samples=100_000,
        val_samples=20_000,
        test_samples=20_000,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_workers = num_workers
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples

    def setup(self, stage=None):
        # Only use the first N rows from each split
        self.train_ds = AmazonReviewDataset(
            self.data_dir / "train.csv",
            self.tokenizer_name,
            self.max_len,
            n_samples=self.train_samples
        )
        self.val_ds = AmazonReviewDataset(
            self.data_dir / "train.csv",  # Or use "val.csv" if available
            self.tokenizer_name,
            self.max_len,
            n_samples=self.val_samples
        )
        self.test_ds = AmazonReviewDataset(
            self.data_dir / "test.csv",
            self.tokenizer_name,
            self.max_len,
            n_samples=self.test_samples
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
