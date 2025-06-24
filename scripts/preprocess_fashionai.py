import pandas as pd
from pathlib import Path
from tqdm import tqdm

RAW = Path('/data_vault/COmparative_Study_of_Multimodal_Represenations/data/raw/fashionai')
PROCESSED = Path('/data_vault/COmparative_Study_of_Multimodal_Represenations/data/processed/fashionai')
PROCESSED.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW / 'data.csv')
img_dir = RAW / 'data'
# Filter to keep only rows with actual image files
df['image_path'] = df['image'].apply(lambda x: str(img_dir / x) if (img_dir / x).exists() else None)
df = df.dropna(subset=['image_path'])
# Basic clean text fields
df['description'] = df['description'].fillna('').astype(str).str.strip().str.lower()
df['label'] = df['category'].astype(str)
df_out = df[['image_path', 'description', 'label']]
# Split (e.g. 80/20 train/val split)
df_out = df_out.sample(frac=1, random_state=42).reset_index(drop=True)
n_train = int(0.8 * len(df_out))
df_out.iloc[:n_train].to_csv(PROCESSED / 'train.csv', index=False)
df_out.iloc[n_train:].to_csv(PROCESSED / 'val.csv', index=False)
print("FashionAI preprocessing done!")
