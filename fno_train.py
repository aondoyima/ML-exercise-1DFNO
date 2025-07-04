# fno_train.py

import os, glob, json, random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau  # optional
from fno_dataset import FNODataset1DWithParams
from fno_model import FNO1D
import fno_utils

# Config
epochs = 100
batch_size = 4
learning_rate = 1e-4
schedule = True
save_every = 1
rollout_steps = 1
model_save_path = 'checkpoints'
log_dir = 'runs/fno_experiment'

os.makedirs(model_save_path, exist_ok=True)
writer = SummaryWriter(log_dir)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Data setup
all_sim_dirs = glob.glob('./training_data/**/', recursive=True)
all_sim_dirs = [d for d in all_sim_dirs if os.path.exists(os.path.join(d, 'frame0.p'))]
random.seed(42)
random.shuffle(all_sim_dirs)

split_idx = int(0.8 * len(all_sim_dirs))
train_dirs = all_sim_dirs[:split_idx]
test_dirs = all_sim_dirs[split_idx:]

pre_dataset = FNODataset1DWithParams(train_dirs)
mean, std = fno_utils.compute_mean_std(pre_dataset)
torch.save({'mean': mean, 'std': std}, 'normalization_stats.pt')

train_dataset = FNODataset1DWithParams(train_dirs, mean=mean.numpy(), std=std.numpy(), rollout_steps=rollout_steps)
test_dataset = FNODataset1DWithParams(test_dirs, mean=mean.numpy(), std=std.numpy(), rollout_steps=rollout_steps)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model
num_params = len(train_dataset.param_names)
model = FNO1D(modes=32, width=64, in_channels=3 + num_params, out_channels=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if schedule:
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

# Save config
config = {
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "rollout_steps": rollout_steps,
    "model_params": {
        "modes": 32,
        "width": 64,
        "in_channels": 3 + num_params,
        "out_channels": 3
    },
    "device": str(device)
}
with open(os.path.join(log_dir, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

# Training
train_losses, val_losses, grad_norms = [], [], []

for epoch in range(1, epochs + 1):
    model.train()
    total_train_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False)

    for u0, theta, targets in pbar:
        u0, theta, targets = u0.to(device), theta.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = 0.0
        u_t = u0

        for step in range(rollout_steps):
            pred = model(u_t, theta)
            true = targets[:, step]
            loss += F.mse_loss(pred, true)
            u_t = pred.detach()

        loss /= rollout_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_train_loss += loss.item() * u0.size(0)
        pbar.set_postfix({"batch_loss": loss.item()})

    avg_train_loss = total_train_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)
    writer.add_scalar("Loss/Train", avg_train_loss, epoch)

    # Grad norm
    grad_norm = torch.sqrt(sum((p.grad**2).sum() for p in model.parameters() if p.grad is not None)).item()
    grad_norms.append(grad_norm)
    writer.add_scalar("GradNorm", grad_norm, epoch)

    # Learning rate
    writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], epoch)

    # Validation
    val_loss = fno_utils.validate(model, test_loader, device, rollout_steps=rollout_steps)
    val_losses.append(val_loss)
    writer.add_scalar("Loss/Val", val_loss, epoch)

    if schedule:
        scheduler.step(val_loss)

    print(f"[Epoch {epoch:03d}] Train Loss: {avg_train_loss:.4e} | Val Loss: {val_loss:.4e}")

    if epoch % save_every == 0 or epoch == 1:
        torch.save(model.state_dict(), os.path.join(model_save_path, f'model_epoch_{epoch}.pt'))

# Save final diagnostic data
np.savez(os.path.join(log_dir, 'loss_data.npz'),
         train_losses=np.array(train_losses),
         val_losses=np.array(val_losses),
         grad_norms=np.array(grad_norms))

# Loss curve
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', linewidth=2)
plt.plot(val_losses, label='Validation Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss Over Epochs')
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(log_dir, 'loss_plot.png'), dpi=300)
plt.close()

writer.close()
print("Training complete.")
