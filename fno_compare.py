import numpy as np
import torch
from fno_model import FNO1D
import fno_utils
import pickle

# Select device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") 

# Params
alpha = 0.43
num_params = 1

# Load model and stats
model = FNO1D(modes=32, width=64, in_channels=3 + num_params, out_channels=3)
model.load_state_dict(torch.load("checkpoints/model_epoch_100.pt"))
model = model.to(device) 
stats = torch.load("normalization_stats.pt")
mean, std = stats['mean'], stats['std']

# Initial condition
ic = pickle.load(open('test_ic.p','rb'))
u0_np = np.stack([ic['psi1'], ic['psi2'], ic['h']], axis=0)  

# Parameters
theta_np = np.array([alpha], dtype=np.float32) 

# Roll out dynamics
t_steps = 5000 
u_torch = fno_utils.predict_trajectory_from_numpy(model, u0_np, theta_np, t_steps, mean, std, device)

# Save in same format as ground truth simulation
N = len(ic['psi1'])
L = 40*np.pi
x = np.linspace(-L/2, L/2, N, endpoint=False)
dt = 2 # this is hardcoded but can be read from the data - it's the time interval at which I save the simulations
pred_dir = f'./data_pred/alpha={alpha}'
truth_dir = f'./data_truth/alpha={alpha}'
fno_utils.save_rollout_predictions(u_torch, x, dt, pred_dir)

# Load in right format for plotting
u_pred, x_pred, t_pred = fno_utils.load_rollout_from_frames(pred_dir)
u_truth, x_truth, t_truth = fno_utils.load_rollout_from_frames(truth_dir)

# Plot kymographs
fno_utils.make_kymograph_from_numpy(u_pred, x, t_pred, pred_dir+'/kym_pred.png')
fno_utils.make_kymograph_from_numpy(u_truth, x, t_truth, truth_dir+'/kym_true.png')

# Plot mean square error 
fno_utils.plot_mse_over_time(u_pred, u_truth, f'mse_alpha={alpha}.png', per_field=True)
