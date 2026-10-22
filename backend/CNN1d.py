import torch.nn as nn
import copy
import torch
import numpy as np
from typing import List
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader
from src.helpers import compute_metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau


class CNN1d(nn.Module):
    def __init__(self, in_channels: int, num_classes:int, out_channels: List[int], 
                    kernel_sizes: List[int], pool_sizes: List[int], strides: List[int] = [], 
                    dense_units: List[int] = [], device = None, use_batch_norm = True,
                    dropout_rate: int = 0.4
                ):
        super().__init__()

        assert len(out_channels) == len(kernel_sizes) == len(pool_sizes), \
            "num_filters, kernel_sizes, and pool_sizes must all be the same length"
        if len(strides) < len(pool_sizes):
            strides = strides + pool_sizes[len(strides):]

        layers = []
        
        in_ch = in_channels
        for i in range(len(out_channels)):
            out_ch = out_channels[i]
            kernel = kernel_sizes[i]
            pool = pool_sizes[i]
            stride = strides[i]
            padding = (kernel-1)//2
            layers.append(nn.Conv1d(in_ch, out_ch, kernel, padding))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_ch))

            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(pool, stride))

            in_ch = out_ch

        self.conv_blocks = nn.Sequential(*layers)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        
        fc_layers = []

        fc_in = in_ch

        for unit in dense_units:
            fc_layers.append(nn.Linear(fc_in, unit))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout_rate))
            
            fc_in = unit
        
        fc_layers.append(nn.Linear(fc_in, num_classes))

        self.fc_head = nn.Sequential(*fc_layers)
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, x):
            x = x.permute(0, 2, 1)
            x = self.conv_blocks(x)
            x = self.global_avg_pool(x)
            x = x.squeeze(-1)    
            x = self.fc_head(x)
            return x
    

    def train_model(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int = 50,
            learning_rate: float = 0.001, patience: int = 5,
            optimizer : Optimizer = Adam, weight_decay: float = 1e-4,
            criterion: nn.Module =  nn.BCEWithLogitsLoss, pos_weight: torch.Tensor = None,
        ):
        optimizer = optimizer(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = criterion(pos_weight= pos_weight)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
        history          = []
        best_auc         = -float('inf')
        best_val_loss    = float('inf')
        best_weights     = copy.deepcopy(self.state_dict())
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):

            # ── Training ──────────────────────────────────────────────
            self.train()
            train_losses = []
            train_logits = []
            train_targets = []

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits = self(x_batch)
                loss   = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                train_logits.append(logits.detach().cpu().numpy())
                train_targets.append(y_batch.detach().cpu().numpy())

            train_metrics         = compute_metrics(np.vstack(train_logits), np.vstack(train_targets))
            train_metrics['loss'] = float(np.mean(train_losses))

            # ── Validation ────────────────────────────────────────────
            self.eval()
            val_losses  = []
            val_logits  = []
            val_targets = []

            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    logits  = self(x_batch)
                    loss    = criterion(logits, y_batch)

                    val_losses.append(loss.item())
                    val_logits.append(logits.cpu().numpy())
                    val_targets.append(y_batch.cpu().numpy())

            val_metrics         = compute_metrics(np.vstack(val_logits), np.vstack(val_targets))
            val_metrics['loss'] = float(np.mean(val_losses))
            if np.isfinite(val_metrics['macro_auc']):
                scheduler.step(val_metrics['macro_auc'])

            row = {
                'epoch': epoch,
                'train': train_metrics,
                'val':   val_metrics,
            }
            history.append(row)

            print(
                f"Epoch {epoch:03d} | "
                f"train loss {train_metrics['loss']:.4f} auc {train_metrics['macro_auc']:.4f} | "
                f"val loss {val_metrics['loss']:.4f} auc {val_metrics['macro_auc']:.4f}"
            )

            # ── Early stopping on val macro AUC ───────────────────────
            val_auc = val_metrics['macro_auc']
            val_loss = val_metrics['loss']
            if (np.isfinite(val_auc) and val_auc > best_auc) or (not np.isfinite(val_auc) and val_loss < best_val_loss):
                if np.isfinite(val_auc):
                    best_auc = val_auc
                best_val_loss    = val_loss
                best_weights     = copy.deepcopy(self.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    if np.isfinite(best_auc):
                        print(f"\nEarly stopping at epoch {epoch}. Best val AUC: {best_auc:.4f}")
                    else:
                        print(f"\nEarly stopping at epoch {epoch}. Best val loss: {best_val_loss:.4f}")
                    break

        self.load_state_dict(best_weights)
        return history

    def predict(self, test_loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        self.eval()
        all_logits  = []
        all_targets = []

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(self.device, non_blocking=True)
                logits  = self(x_batch)
                all_logits.append(logits.detach().cpu().numpy())
                all_targets.append(y_batch.detach().cpu().numpy())

        return np.vstack(all_logits), np.vstack(all_targets)
