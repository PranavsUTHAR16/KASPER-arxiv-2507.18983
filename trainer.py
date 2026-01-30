import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from losses import composite_loss
from torch.optim.lr_scheduler import LambdaLR

class ForecasterTrainer:
    def __init__(self, model, lr=0.0005, weight_decay=1e-5, patience=40):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # AdamW as per paper Table I (reduced LR for stability)
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
        
        # Temperature Annealing parameters (Fix 1) - SLOWER decay
        self.initial_temperature = 2.0      # Increased from 1.0
        self.final_temperature = 0.5        # Increased from 0.3
        self.temperature_decay = 0.98       # Slower from 0.95
        
        self.patience = patience
        self.best_val_loss = float('inf')
        self.best_healthy_val_loss = float('inf')  # NEW: Track best HEALTHY model
        self.counter = 0
        
        # NEW: Regime health tracking
        self.regime_health_history = []
        self.best_healthy_epoch = -1

    def check_regime_health(self, regime_probs):
        """
        Check if regime distribution is healthy (no collapse).
        Returns (is_healthy, health_score, regime_dist)
        """
        regime_assignments = regime_probs.argmax(dim=1)
        n_regimes = regime_probs.size(1)
        regime_counts = regime_assignments.bincount(minlength=n_regimes)
        regime_dist = (regime_counts / regime_counts.sum()).cpu().numpy()
        
        # Health criteria:
        # 1. All regimes have at least 10% of samples
        min_regime_pct = regime_dist.min()
        
        # 2. No regime has more than 60% (indicating dominance)
        max_regime_pct = regime_dist.max()
        
        # 3. Entropy is high (diverse distribution)
        regime_entropy = -np.sum(regime_dist * np.log(regime_dist + 1e-8))
        max_entropy = np.log(n_regimes)
        entropy_ratio = regime_entropy / max_entropy
        
        # Health score (0-1, higher is better)
        health_score = (
            0.4 * min_regime_pct / 0.33 +           # Reward balanced minimum
            0.3 * (1.0 - max_regime_pct / 1.0) +    # Penalize dominance
            0.3 * entropy_ratio                      # Reward high entropy
        )
        
        # Regime is healthy if:
        # - Each regime has at least 10% samples
        # - No regime has more than 60% samples  
        # - Entropy ratio > 0.7 (fairly diverse)
        is_healthy = (
            min_regime_pct >= 0.10 and 
            max_regime_pct <= 0.60 and
            entropy_ratio >= 0.7
        )
        
        return is_healthy, health_score, regime_dist

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        epoch_losses = []
        epoch_details = []
        all_regime_probs = []
        
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
            
            # Collect regime probs for health check
            all_regime_probs.append(regime_probs.detach())
            
            loss, loss_details = composite_loss(
                y_pred, batch_y, embeddings, regime_probs, self.model
            )
            
            loss.backward()
            
            # Gradient clipping (0.5 as per paper)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            
            self.optimizer.step()
            epoch_losses.append(loss.item())
            epoch_details.append(loss_details)
            
        # Check regime health
        all_regime_probs = torch.cat(all_regime_probs, dim=0)
        is_healthy, health_score, regime_dist = self.check_regime_health(all_regime_probs)
        
        # Logging
        health_icon = "✅" if is_healthy else "⚠️"
        print(f"  Regime dist: {regime_dist} | Temp: {current_temp:.3f} {health_icon}")
        print(f"  Health score: {health_score:.3f} | Healthy: {is_healthy}")
        
        if epoch_details:
            avg_losses = {k: np.mean([d[k] for d in epoch_details]) for k in epoch_details[0]}
            loss_str = " ".join([f"{k}:{v:.3f}" for k, v in avg_losses.items()])
            print(f"  Losses -> {loss_str}")
        
        return np.mean(epoch_losses), is_healthy, health_score, regime_dist

    def validate(self, val_loader):
        self.model.eval()
        val_losses = []
        all_regime_probs = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                y_pred, regime_probs, embeddings = self.model(batch_x)
                all_regime_probs.append(regime_probs)
                
                loss, _ = composite_loss(
                    y_pred, batch_y, embeddings, regime_probs, self.model
                )
                val_losses.append(loss.item())
        
        # Check validation regime health
        all_regime_probs = torch.cat(all_regime_probs, dim=0)
        is_healthy, health_score, regime_dist = self.check_regime_health(all_regime_probs)
        
        return np.mean(val_losses), is_healthy, health_score

    def fit(self, X_train, y_train, X_val, y_val, epochs=200, batch_size=64):
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val)
        
        # CRITICAL BUG FIX: Enable shuffling for training data
        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t), 
            batch_size=batch_size, 
            shuffle=True  # ✅ FIXED
        )
        val_loader = DataLoader(
            TensorDataset(X_val_t, y_val_t), 
            batch_size=batch_size, 
            shuffle=False
        )
        
        print("\n🔍 Monitoring regime health during training...")
        print("✅ = Healthy regimes | ⚠️ = Regime issues detected\n")
        
        for epoch in range(epochs):
            train_loss, train_healthy, train_health_score, regime_dist = self.train_epoch(train_loader, epoch)
            val_loss, val_healthy, val_health_score = self.validate(val_loader)
            
            # Store regime health history
            self.regime_health_history.append({
                'epoch': epoch,
                'val_loss': val_loss,
                'is_healthy': val_healthy,
                'health_score': val_health_score,
                'regime_dist': regime_dist
            })
            
            # Update sparsity mask - ONLY after epoch 50 with gentler threshold
            if (epoch + 1) % 10 == 0 and epoch >= 50:
                self.model.update_sparsity(threshold=0.001)  # Gentler than 0.01
                print("  🔧 Updated sparsity mask")
                
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
            
            # Update schedulers (Fix 5)
            if epoch < 10:
                self.warmup_scheduler.step()
            else:
                self.plateau_scheduler.step(val_loss)
            
            health_status = "✅ HEALTHY" if val_healthy else "⚠️ UNHEALTHY"
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | {health_status}")
            
            # IMPROVED EARLY STOPPING: Consider BOTH loss and regime health
            # Strategy 1: Save if validation loss improves AND regimes are healthy
            if val_loss < self.best_healthy_val_loss and val_healthy:
                self.best_healthy_val_loss = val_loss
                self.best_healthy_epoch = epoch
                self.counter = 0
                torch.save(self.model.state_dict(), 'best_kasper_healthy.pth')
                print(f"  💾 Saved HEALTHY model (epoch {epoch+1}, loss={val_loss:.6f})")
            
            # Strategy 2: Always track absolute best loss (even if unhealthy) as fallback
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_kasper.pth')
            else:
                self.counter += 1
                
            # Early stopping with warning
            if self.counter >= self.patience:
                print(f"\n⚠️ Early stopping triggered at epoch {epoch+1}")
                if self.best_healthy_epoch >= 0:
                    print(f"✅ Will load best HEALTHY model from epoch {self.best_healthy_epoch+1}")
                else:
                    print("⚠️ No healthy model found! Using best loss model (may have regime collapse)")
                break
        
        # Load best model
        print("\n📊 Training complete. Loading best model...")
        
        # Prefer healthy model if exists
        import os
        if os.path.exists('best_kasper_healthy.pth'):
            print(f"✅ Loading HEALTHY model from epoch {self.best_healthy_epoch+1}")
            if torch.cuda.is_available():
                self.model.load_state_dict(torch.load('best_kasper_healthy.pth'))
            else:
                self.model.load_state_dict(torch.load('best_kasper_healthy.pth', map_location=torch.device('cpu')))
        else:
            print("⚠️ No healthy model saved. Loading best loss model (may have collapsed regimes)")
            if torch.cuda.is_available():
                self.model.load_state_dict(torch.load('best_kasper.pth'))
            else:
                self.model.load_state_dict(torch.load('best_kasper.pth', map_location=torch.device('cpu')))
        
        # Print regime health summary
        print("\n📈 Regime Health History:")
        healthy_epochs = [h for h in self.regime_health_history if h['is_healthy']]
        if healthy_epochs:
            print(f"  Healthy epochs: {len(healthy_epochs)}/{len(self.regime_health_history)}")
            print(f"  Best healthy epoch: {self.best_healthy_epoch+1} (loss={self.best_healthy_val_loss:.6f})")
        else:
            print("  ⚠️ NO healthy epochs found during training!")
            print("  This indicates severe over-regularization or insufficient model capacity.")
            print("  Recommendation: Increase lambda_balance to 0.5+ or reduce lambda_l1 to 0.0")
        
        return self.model
