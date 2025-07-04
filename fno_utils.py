# fno_utils.py

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def compute_mean_std(dataset):
    sum_ = 0.0
    sum_sq = 0.0
    count = 0

    for u_t, _, _ in dataset:
        sum_ += u_t.sum(dim=1)  # sum over X, shape: (C,)
        sum_sq += (u_t ** 2).sum(dim=1)
        count += u_t.shape[1]   # number of spatial points

    mean = sum_ / count
    std = (sum_sq / count - mean**2).sqrt()

    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean, std

def denormalize(u, mean, std):
    # u: (T, C, X)
    # mean, std: (C,) or (C,1)
    if mean.dim() == 1:
        mean = mean.view(1, -1, 1)  # shape (1, C, 1)
    if std.dim() == 1:
        std = std.view(1, -1, 1)
    return u * std + mean

def predict_trajectory_from_numpy(model, u0_np, theta_np, T, mean, std, device):  
    u0 = torch.tensor(u0_np, dtype=torch.float32)
    theta = torch.tensor(theta_np, dtype=torch.float32)

    # Normalize input
    if mean.dim() == 1:
        mean_n = mean.unsqueeze(1)  # from (3,) to (3,1)
    if std.dim() == 1:
        std_n = std.unsqueeze(1)
    u0 = (u0 - mean_n) / std_n

    # Run rollout
    model.eval()
    with torch.no_grad():
        traj = rollout_prediction(model, u0, theta, T, device=device)

    # Denormalize full trajectory
    pred_d = denormalize(traj, mean, std)
    return pred_d  # (T+1, C, X)

def rollout_prediction(model, u0, theta, T, device):
    u = u0.unsqueeze(0).to(device)       # (1, C, X)
    theta = theta.unsqueeze(0).to(device)  # (1, P)
    traj = [u0.cpu()]  # store initial condition

    for _ in range(T):
        with torch.no_grad():
            u_next = model(u, theta)     # (1, C, X)
        traj.append(u_next.squeeze(0).cpu())
        u = u_next

    return torch.stack(traj)  # shape (T+1, C, X)

def validate(model, loader, device, rollout_steps=5):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for u0, theta, targets in loader:
            u0, theta, targets = u0.to(device), theta.to(device), targets.to(device)
            loss = 0.0
            u_t = u0

            for step in range(rollout_steps):
                pred = model(u_t, theta)
                true = targets[:, step]
                loss += F.mse_loss(pred, true)
                u_t = pred  # feed back prediction

            loss = loss / rollout_steps
            total_loss += loss.item() * u0.size(0)

    return total_loss / len(loader.dataset)

def save_rollout_predictions(u_pred, x, dt, save_dir):
    import os, pickle
    os.makedirs(save_dir, exist_ok=True)
    u_pred = u_pred.detach().cpu().numpy()
    T, C, X = u_pred.shape

    for t_idx in range(T):
        data = {
            't': t_idx * dt,
            'x': x,
            'psi1': u_pred[t_idx, 0],
            'psi2': u_pred[t_idx, 1],
            'h': u_pred[t_idx, 2],
        }
        with open(os.path.join(save_dir, f'frame{t_idx}.p'), 'wb') as f:
            pickle.dump(data, f)

def load_rollout_from_frames(folder):
    import os, pickle, numpy as np
    files = sorted([f for f in os.listdir(folder) if f.startswith('frame')], key=lambda f: int(f[5:-2]))
    u_list, t_list = [], []
    for fname in files:
        with open(os.path.join(folder, fname), 'rb') as f:
            data = pickle.load(f)
            if 'x' not in locals():
                x = np.array(data['x'])
            t_list.append(data['t'])
            u_list.append([
                np.array(data['psi1']),
                np.array(data['psi2']),
                np.array(data['h']),
            ])
    return np.stack(u_list), x, np.array(t_list)

def make_kymograph_from_numpy(u, x, t, figname):
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    T, C, X = u.shape
    field_order = [2, 0, 1]  # h, psi1, psi2
    names = [r'$h(x,t)$', r'$\psi_1(x,t)$', r'$\psi_2(x,t)$']
    cmaps = ['Spectral', 'Blues', 'Blues']

    fig = plt.figure(figsize=(9, 4))
    gs = gridspec.GridSpec(3, 2, width_ratios=[20, 1])

    for i, ch in enumerate(field_order):
        ax = fig.add_subplot(gs[i, 0])
        cax = fig.add_subplot(gs[i, 1])
        im = ax.pcolormesh(t, x, u[:, ch, :].T, cmap=cmaps[i], shading='nearest',
                           vmin=u[:, ch, :].min(), vmax=u[:, ch, :].max(), rasterized=True)
        ax.set_ylabel(r'$x$')
        if i == 2:
            ax.set_xlabel(r'$t$')
        else:
            ax.tick_params(labelbottom=False)
        ax.text(0.05, 1.02, names[i], transform=ax.transAxes, fontsize=9, va='bottom')
        plt.colorbar(im, cax=cax)
    
    plt.tight_layout()
    plt.savefig(figname, dpi=400)
    plt.close(fig)

def plot_mse_over_time(u_pred, u_true, save_path, per_field=True):
    import matplotlib.pyplot as plt
    import numpy as np
    T, C, X = u_pred.shape

    if per_field:
        mse = ((u_pred - u_true) ** 2).mean(axis=2)  # shape: (T, C)
        plt.figure(figsize=(7, 4))
        field_labels = [r'$\psi_1$', r'$\psi_2$', r'$h$']
        for i in range(C):
            plt.plot(mse[:, i], label=f'MSE {field_labels[i]}')
    else:
        mse = ((u_pred - u_true) ** 2).mean(axis=(1, 2))
        plt.figure(figsize=(7, 4))
        plt.plot(mse, label='MSE over time')

    plt.xlabel('Time step')
    plt.ylabel('Mean Squared Error')
    plt.yscale('log')
    plt.grid(True)
    plt.title("Prediction Error Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()