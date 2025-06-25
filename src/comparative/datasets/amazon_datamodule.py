import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pytorch_lightning as pl
from pathlib import Path

class AmazonDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=128):
        df = pd.read_csv(csv_path)
        self.texts = df['full_text'].astype(str).tolist()
        self.labels = df['label'].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

class AmazonDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, max_len=128, model_name='distilbert-base-uncased', num_workers=8):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.max_len = max_len
        self.model_name = model_name
        self.num_workers = num_workers

    def setup(self, stage=None):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.train_ds = AmazonDataset(self.data_dir / 'train.csv', tokenizer, self.max_len)
        self.val_ds = AmazonDataset(self.data_dir / 'test.csv', tokenizer, self.max_len)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)
