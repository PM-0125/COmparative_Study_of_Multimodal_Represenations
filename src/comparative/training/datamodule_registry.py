# src/comparative/training/datamodule_registry.py

from comparative.datasets.amazon_datamodule import AmazonDataModule
from comparative.datasets.fashion_datamodule import FashionAIDatamodule
from comparative.datasets.movielens_datamodule import MovieLensDataModule

def get_datamodule(data_cfg):
    """Return the correct DataModule based on config or string name."""
    if isinstance(data_cfg, dict):
        data_name = data_cfg.get("_target_") or data_cfg.get("name", "")
    else:
        data_name = str(data_cfg)

    if "amazon" in data_name.lower():
        return AmazonDataModule(**data_cfg)
    elif "fashion" in data_name.lower():
        return FashionAIDatamodule(**data_cfg)
    elif "movielens" in data_name.lower():
        return MovieLensDataModule(**data_cfg)
    else:
        raise ValueError(f"Unknown dataset/datamodule: {data_name}")
