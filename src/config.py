from dataclasses import dataclass

@dataclass
class Config():
    case: str = 'base'
    # Model Def
    d_in: int = 1 # not including t
    d_out: int = 1
    d_hidden: int = 256
    num_layers: int = 5
    activation: str = 'swish'
    sine_omega0: float = 30.0
    four_emb: bool = False
    emb_dim: int = 256
    emb_scale: float = 1
    skip_conn: bool = False
    save_layers: tuple = (1,3,5,7)
    skip_layers: tuple = (3,5,7,9)

    # Domain Def
    x_range: tuple = ((-1,1),)
    t_range: tuple = (0,1)

    # Training Def
    batch_pde: int = 256
    batch_ic: int = 256
    batch_traj: int = 64
    optim: str = 'adam'
    lr: float = 1e-3
    iter: int = 250000
    loss_method: str = 'pinns'
    additional_losses: bool = False

    # Adaptive PDE sampling (RAD)
    use_rad_sampling: bool = False
    rad_k: float = 1.0
    rad_c: float = 1.0
    rad_candidate_size: int = 32768
    rad_refresh_interval: int = 1000
    rad_warmup_iters: int = 1000
    rad_residual_batch_size: int = 4096
    rad_residual_eps: float = 1e-12
    
    # PINNS Loss Def
    pde_scale: float = 1
    ic_scale: float = 10

    # Trajectory rollout def
    traj_len: int = 50
    delta_t: float = 2e-2

    # HJI Def
    hard_constraint: bool = True
    terminal_hard_constraint: bool = False  # Enforce V=l at terminal time T using normalized time factor
    alternative_tc: bool = False  # Alternative terminal constraint using phi_T instead of time factor
    hard_constraint_type: str = 'quadratic'  # 'quadratic', 'softplus', 'swish', 'elu'
    term_grad_loss: bool = True  # Include gradient loss at terminal time
    sigma_noise: float = 0.1
    f1_map_path: str = "src/assets/f1tenth/F1_map_obstaclemap.mat"
    f1_use_rejection_sampling: bool = True
    f1_rejection_train_l_min: float = -1.0
    f1_rejection_eval_l_min: float = 0.0
    f1_rejection_oversample_factor: float = 2.0
    f1_rejection_max_rounds: int = 64
 
    #extras
    save_to_wandb: bool = False
    checkpointing: bool = False
    periodic: bool = False
    periodic_idx: tuple = (1,)
    input_normalization: bool = False
    parallel: bool = False
    batch_grad: bool = False
    random_sample: bool = True
    auto_close: bool = True
    stop_grad: bool = False
    bound_rollout_states: bool = False  # Wrap periodic states + clip non-periodic states during rollout integration
    smooth_control: bool = False
    smooth_control_scope: str = "training_only"  # "off", "training_only", "all"
    smooth_control_rho: float = 0.05
    smooth_control_map: str = "sqrt"  # "sqrt" (C-infinity), "c1"

    # Safety Evaluation Metric (HJI only)
    safety_eval: bool = False
    num_safety_rollouts: int = 10000
    safety_rollout_batch_size: int = 1000
    safety_boundary_threshold: float = 0.01  # Points within this distance from boundary (V ≈ 0) are excluded

    def get_train_config(self):
        return TrainConfig(batch_pde=self.batch_pde,
                           batch_ic=self.batch_ic,
                           optim=self.optim,
                           lr=self.lr,
                           iter=self.iter,
                           loss_method=self.loss_method,
                           use_rad_sampling=self.use_rad_sampling,
                           rad_k=self.rad_k,
                           rad_c=self.rad_c,
                           rad_candidate_size=self.rad_candidate_size,
                           rad_refresh_interval=self.rad_refresh_interval,
                           rad_warmup_iters=self.rad_warmup_iters,
                           rad_residual_batch_size=self.rad_residual_batch_size,
                           rad_residual_eps=self.rad_residual_eps,
                           pde_scale=self.pde_scale,
                           ic_scale=self.ic_scale,
                           traj_len=self.traj_len,
                           delta_t=self.delta_t,
                           bound_rollout_states=self.bound_rollout_states)

@dataclass
class TrainConfig():
    # Training Def
    batch_pde: int = 256
    batch_ic: int = 256
    optim: str = 'adam'
    lr: float = 1e-3
    iter: int = 250000
    loss_method: str = 'pinns'

    # Adaptive PDE sampling (RAD)
    use_rad_sampling: bool = False
    rad_k: float = 1.0
    rad_c: float = 1.0
    rad_candidate_size: int = 16384
    rad_refresh_interval: int = 200
    rad_warmup_iters: int = 1000
    rad_residual_batch_size: int = 4096
    rad_residual_eps: float = 1e-12

    # PINNS Loss Def
    pde_scale: float = 1
    ic_scale: float = 1

    # Trajectory rollout def
    traj_len: int = 50
    delta_t: float = 1e-2
    bound_rollout_states: bool = False
