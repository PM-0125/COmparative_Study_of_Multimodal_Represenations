import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

PROCESSED = Path('/data_vault/COmparative_Study_of_Multimodal_Represenations/data/processed/amazon')

df = pd.read_csv(PROCESSED / 'train.csv')
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])

train_df.to_csv(PROCESSED / 'train.csv', index=False)
val_df.to_csv(PROCESSED / 'val.csv', index=False)

print("Created train/val split for Amazon.")
