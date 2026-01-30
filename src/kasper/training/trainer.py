import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from .losses import composite_loss
from torch.optim.lr_scheduler import LambdaLR

class ForecasterTrainer:
    def __init__(self, model, lr=0.001, weight_decay=1e-5, patience=15):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # AdamW as per paper Table I
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Add warmup scheduler (Fix 5)
        def lr_lambda(epoch):
            warmup_epochs = 10
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                return 1.0
        
        self.warmup_scheduler = LambdaLR(self.optimizer, lr_lambda)
        
        # Scheduler: factor 0.7, patience 7 as per Table I
        self.plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.7, patience=7
        )
        
        # Temperature Annealing parameters (Diagnosis Fix 2)
        self.initial_temperature = 2.0
        self.final_temperature = 0.5
        self.temperature_decay = 0.98
        
        self.patience = patience
        self.best_val_loss = float('inf')
        self.counter = 0

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        epoch_losses = []
        epoch_details = []
        regime_counts = torch.zeros(self.model.n_regimes)
        
        # CRITICAL: Anneal temperature to prevent regime collapse (Fix 1)
        current_temp = max(
            self.final_temperature,
            self.initial_temperature * (self.temperature_decay ** epoch)
        )
        self.model.temperature = current_temp
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            self.optimizer.zero_grad()
            
            y_pred, regime_probs, embeddings = self.model(batch_x)
            
            # Track regime distribution
            regime_counts += regime_probs.argmax(dim=1).bincount(minlength=self.model.n_regimes).cpu()
            
            loss, loss_details = composite_loss(
                y_pred, batch_y, embeddings, regime_probs, self.model
            )
            
            loss.backward()
            
            # Gradient clipping (0.5 as per paper)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            
            self.optimizer.step()
            epoch_losses.append(loss.item())
            epoch_details.append(loss_details)
            
        # Logging regime balance
        regime_dist = (regime_counts / regime_counts.sum()).numpy()
        
        if epoch_details:
            avg_losses = {k: np.mean([d[k] for d in epoch_details]) for k in epoch_details[0]}
            loss_str = " ".join([f"{k}:{v:.3f}" for k, v in avg_losses.items()])
            print(f"  Regime dist: {regime_dist} | Temp: {current_temp:.3f}")
            print(f"  Losses -> {loss_str}")
        
        return np.mean(epoch_losses)

    def validate(self, val_loader):
        self.model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                y_pred, regime_probs, embeddings = self.model(batch_x)
                
                loss, _ = composite_loss(
                    y_pred, batch_y, embeddings, regime_probs, self.model
                )
                val_losses.append(loss.item())
                
        return np.mean(val_losses)

    def fit(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val)
        
        # CRITICAL BUG FIX: Enable shuffling for training data
        # Previous: shuffle=False caused temporal overfitting and negative R²
        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t), 
            batch_size=batch_size, 
            shuffle=True  # ✅ FIXED: Was False, now True
        )
        val_loader = DataLoader(
            TensorDataset(X_val_t, y_val_t), 
            batch_size=batch_size, 
            shuffle=False  # Validation doesn't need shuffling
        )
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader)
            
            # Update sparsity mask every 10 epochs as per paper (Diagnosis Fix 3)
            if (epoch + 1) % 10 == 0 and epoch > 50:
                self.model.update_sparsity(threshold=0.001)
                
            # Refine forecasting knots after Layer 1 embeddings stabilize (Epoch 5)
            if epoch == 5:
                print("  Refining forecasting knots using quantile initialization...")
                self.model.eval()
                with torch.no_grad():
                    all_x = train_loader.dataset.tensors[0].to(self.device)
                    _, r_probs, embeddings = self.model(all_x)
                    if hasattr(self.model, 'models'):
                        for m in self.model.models:
                            m.forecasting.fit_knots(embeddings, r_probs)
                    else:
                        self.model.forecasting.fit_knots(embeddings, r_probs)
                self.model.train()
            
            # Grid extension (Coarse-to-fine refinement) after stable knots found (Epoch 20)
            if epoch == 20:
                print("  Performing grid extension (10 -> 20 basis functions)...")
                self.model.eval()
                with torch.no_grad():
                    all_x = train_loader.dataset.tensors[0].to(self.device)
                    _, _, embeddings = self.model(all_x)
                    if hasattr(self.model, 'models'):
                        for m in self.model.models:
                            m.forecasting.grid_extension(20, h_samples=embeddings)
                    else:
                        self.model.forecasting.grid_extension(20, h_samples=embeddings)
                self.model.train()
                # Update optimizer to include new parameters from grid extension
                self.optimizer = optim.AdamW(self.model.parameters(), lr=self.optimizer.param_groups[0]['lr'], weight_decay=1e-5)
                # Re-wrap warmup scheduler for consistency if needed, but we're past 10 epochs
                # Save as new best to ensure shape matches on subsequent loads
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_kasper.pth')
                
            # Update schedulers (Fix 5)
            if epoch < 10:
                self.warmup_scheduler.step()
            else:
                self.plateau_scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.counter = 0
                torch.save(self.model.state_dict(), 'best_kasper.pth')
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print("Early stopping triggered")
                    break
        
        # Load best model
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load('best_kasper.pth'))
        else:
            self.model.load_state_dict(torch.load('best_kasper.pth', map_location=torch.device('cpu')))
        return self.model
