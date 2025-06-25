import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer

class EarlyFusionClassifier(pl.LightningModule):
    def __init__(self, model_name: str, n_classes: int, lr: float = 2e-5, wd: float = 0.01):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(self.encoder.config.hidden_size, n_classes)
        self.lr = lr
        self.wd = wd
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        out = self.dropout(pooled)
        return self.head(out)

    def training_step(self, batch, batch_idx):
        x, mask, y = batch['input_ids'], batch['attention_mask'], batch['label']
        logits = self(x, mask)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, mask, y = batch['input_ids'], batch['attention_mask'], batch['label']
        logits = self(x, mask)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer
