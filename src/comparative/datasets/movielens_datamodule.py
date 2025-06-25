# src/comparative/datasets/movielens_datamodule.py

import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class MovieLensDataset(Dataset):
    def __init__(self, csv_path, user2idx_path, movie2idx_path, max_len=128):
        df = pd.read_csv(csv_path)
        self.users = df['user_idx'].values
        self.movies = df['movie_idx'].values
        self.ratings = df['rating'].values
        self.titles = df['title'].fillna("").astype(str).tolist()
        self.genres = df['genres'].fillna("").astype(str).tolist()
        self.tags = df['tag'].fillna("").astype(str).tolist()
        # For downstream use, you might want to concatenate title+genre+tag as a text field
        self.max_len = max_len
    def __len__(self):
        return len(self.users)
    def __getitem__(self, idx):
        # For simplicity, returning indices and features; models decide what to use
        return {
            'user_idx': int(self.users[idx]),
            'movie_idx': int(self.movies[idx]),
            'rating': float(self.ratings[idx]),
            'text': f"{self.titles[idx]}. {self.genres[idx]}. {self.tags[idx]}",
        }

class MovieLensDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=64, max_len=128, num_workers=12):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_workers = num_workers
    def setup(self, stage=None):
        self.train_ds = MovieLensDataset(self.data_dir / "train.csv", self.data_dir / "user2idx.csv", self.data_dir / "movie2idx.csv", self.max_len)
        self.val_ds = MovieLensDataset(self.data_dir / "val.csv", self.data_dir / "user2idx.csv", self.data_dir / "movie2idx.csv", self.max_len)
        self.test_ds = MovieLensDataset(self.data_dir / "test.csv", self.data_dir / "user2idx.csv", self.data_dir / "movie2idx.csv", self.max_len)
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
