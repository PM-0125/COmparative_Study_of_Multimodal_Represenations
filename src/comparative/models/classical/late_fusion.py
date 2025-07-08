import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel
from torchvision import models
from typing import Optional

class LateFusionClassifier(pl.LightningModule):
    def __init__(
        self,
        text_model_name: Optional[str] = None,
        image_model_name: Optional[str] = None,
        n_classes: int = 2,
        use_text: bool = True,
        use_image: bool = False,
        image_emb_dim: Optional[int] = None,  # <--- NEW!
        use_graph: bool = False,
        use_tabular: bool = False,
        fusion_hidden: int = 256,
        lr: float = 2e-5,
        wd: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Text branch
        if use_text and text_model_name:
            self.text_encoder = AutoModel.from_pretrained(text_model_name)
            self.text_dim = self.text_encoder.config.hidden_size
            self.text_proj = nn.Identity()
        else:
            self.text_dim = 0
            self.text_encoder = None

        # Image branch (raw image through CNN)
        if use_image and image_model_name:
            base_cnn = models.__dict__[image_model_name](pretrained=True)
            if hasattr(base_cnn, "fc") and hasattr(base_cnn.fc, "in_features"):
                image_dim = base_cnn.fc.in_features
                base_cnn.fc = nn.Identity()
            elif hasattr(base_cnn, "classifier") and hasattr(base_cnn.classifier, "in_features"):
                image_dim = base_cnn.classifier.in_features
                base_cnn.classifier = nn.Identity()
            else:
                raise ValueError("Unknown CNN arch for getting image features!")
            self.image_encoder = base_cnn
            self.image_proj = nn.Linear(image_dim, 256)
            self.image_dim = 256
        else:
            self.image_dim = 0
            self.image_encoder = None

        # Image embedding branch (precomputed .npy)
        self.image_emb_dim = image_emb_dim or 0

        # Other modalities
        self.graph_dim = 128 if use_graph else 0
        self.tabular_dim = 32 if use_tabular else 0

        # Fusion layer input size
        input_dim = self.text_dim + self.image_dim + self.image_emb_dim + self.graph_dim + self.tabular_dim
        self.fusion = nn.Linear(input_dim, fusion_hidden)
        self.head = nn.Linear(fusion_hidden, n_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        features = []
        # Text
        if self.text_encoder is not None and "input_ids" in batch and "attention_mask" in batch:
            text_out = self.text_encoder(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            pooled = text_out.last_hidden_state[:, 0, :]
            features.append(self.text_proj(pooled))
        # Raw image (for models using raw image input)
        if self.image_encoder is not None and "image" in batch:
            img_feat = self.image_encoder(batch["image"])
            features.append(self.image_proj(img_feat))
        # Precomputed image embedding
        if self.image_emb_dim > 0 and "image_emb" in batch:
            features.append(batch["image_emb"])
        # Graph
        if self.graph_dim > 0 and "graph" in batch:
            features.append(batch["graph"])
        # Tabular
        if self.tabular_dim > 0 and "tabular" in batch:
            features.append(batch["tabular"])
        if not features:
            raise ValueError("No input modalities found in batch!")
        fused = torch.cat(features, dim=1)
        fused = torch.relu(self.fusion(fused))
        return self.head(fused)

    def training_step(self, batch, batch_idx):
        y = batch["label"]
        logits = self.forward(batch)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch["label"]
        logits = self.forward(batch)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.hparams['lr'], weight_decay=self.hparams['wd']
        )
