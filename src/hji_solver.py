import sys
sys.path.append('..')
sys.path.append('.')
import jax
import optax
from jax import numpy as jnp
import tqdm
import wandb
from model import PINNs
from config import *
from utils import *
from functools import partial
import copy
import os
import json
import pickle
from datetime import datetime
from typing import Callable, Optional, Set

class HJI_Solver():
    """
    Hamilton-Jacobi-Isaacs (HJI) Solver for differential games and reachability problems.
    """
    ### Initialization and Setup ###
    def __init__(self,config: Config):
        self.config = copy.deepcopy(config)
        self.model = self.create_model()
        self.optimizer = self.create_opt()
        self._init_state_projection_cache()

        if self.config.save_to_wandb:
            self.init_wandb()

    def create_model(self):
        return PINNs(self.config)

    def create_opt(self):
        return optax.inject_hyperparams(optax.adam)(learning_rate=self.config.lr)
    
    def init_wandb(self):
        print("Initializing wandb")
        wandb.init(project="s2r",config=vars(self.config))
    
    def init_solver(self,key:Key):
        params = self.init_model(key)
        opt_state = self.init_opt(params)
        num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print(f"Number of parameters: {num_params}")
        if self.config.save_to_wandb:
            wandb.config['# Params'] =  num_params
        return params,opt_state
    
    def init_model(self,key:Key):
        x,t = self.sample_domain(key,self.config.batch_pde)
        return self.model.init(key.newkey(),x,t)
    
    def init_opt(self,params):
        return self.optimizer.init(params)
    
    def get_base_config():
        return Config()
    
    def close(self):
        if self.config.save_to_wandb:
            wandb.finish()

    def wandb_tags(self,tag):
        wandb.run.tags = wandb.run.tags + (tag,)

    def select_loss(self,loss_method):
        match loss_method:
            case 'vipinns' | 'pinns':
                return self.vipinns_loss
            case 'fspinns' | 'fspinnsbatched':
                return self.fspinns_loss
            
        raise ValueError(f"Invalid loss method: {loss_method}")
    
    def get_control_bounds(self):
        """
        Get control bounds for control sampling.
        
        Override this in subclasses to provide problem-specific bounds.
        Returns (u_min, u_max, d_control) where u_min/u_max are arrays or scalars.
        
        Returns:
            u_min: Lower bound(s) for control, shape (d_control,) or scalar
            u_max: Upper bound(s) for control, shape (d_control,) or scalar
            d_control: Control dimension
        """
        # Default: scalar bounds [-1, 1] with d_control=1
        return -1.0, 1.0, 1

    def _init_state_projection_cache(self):
        x_lower = []
        x_upper = []
        for i in range(int(self.config.d_in)):
            x_range = self.config.x_range[i] if i < len(self.config.x_range) else self.config.x_range[-1]
            x_lower.append(float(x_range[0]))
            x_upper.append(float(x_range[1]))

        dtype = jnp.asarray(0.0).dtype
        self._state_lower_bounds = jnp.asarray(x_lower, dtype=dtype)
        self._state_upper_bounds = jnp.asarray(x_upper, dtype=dtype)
        self._state_wrap_widths = jnp.maximum(self._state_upper_bounds - self._state_lower_bounds, 1e-12)

        periodic_mask = jnp.zeros((int(self.config.d_in),), dtype=bool)
        if bool(getattr(self.config, "periodic", False)):
            periodic_idx = tuple(getattr(self.config, "periodic_idx", ()))
            for idx in periodic_idx:
                idx_int = int(idx)
                if 0 <= idx_int < int(self.config.d_in):
                    periodic_mask = periodic_mask.at[idx_int].set(True)
        self._state_periodic_mask = periodic_mask

    def _use_smooth_controls(self, training: bool) -> bool:
        """
        Resolve whether smooth control should be used for the current context.
        """
        if not bool(getattr(self.config, "smooth_control", False)):
            return False

        scope = str(getattr(self.config, "smooth_control_scope", "training_only")).lower()
        if scope == "off":
            return False
        if scope == "training_only":
            return bool(training)
        if scope == "all":
            return True

        raise ValueError(
            f"Invalid smooth_control_scope: {scope}. Expected one of: off, training_only, all."
        )

    def _use_rollout_state_projection(self) -> bool:
        return bool(getattr(self.config, "bound_rollout_states", False))

    def project_state_to_domain(self, x):
        """
        Project rollout states to configured domain bounds.
        Periodic dimensions are wrapped to [low, high); others are clipped.
        """
        if not self._use_rollout_state_projection():
            return x

        x = jnp.asarray(x)
        shape = (1,) * (x.ndim - 1) + (x.shape[-1],)
        lower = jnp.reshape(self._state_lower_bounds, shape).astype(x.dtype)
        upper = jnp.reshape(self._state_upper_bounds, shape).astype(x.dtype)
        widths = jnp.reshape(self._state_wrap_widths, shape).astype(x.dtype)
        periodic_mask = jnp.reshape(self._state_periodic_mask, shape)

        wrapped = lower + jnp.mod(x - lower, widths)
        clipped = jnp.clip(x, lower, upper)
        return jnp.where(periodic_mask, wrapped, clipped)

    def smooth_sign(self, f, rho=None, map_type=None):
        """
        Smooth approximation of sign(f) with output in [-1, 1].
        """
        rho_val = getattr(self.config, "smooth_control_rho", 0.05) if rho is None else rho
        map_name = getattr(self.config, "smooth_control_map", "sqrt") if map_type is None else map_type

        rho_val = jnp.asarray(rho_val, dtype=f.dtype)
        map_name = str(map_name).lower()

        if map_name == "sqrt":
            # C-infinity map: f / sqrt(rho^2 + f^2)
            return f / jnp.sqrt(rho_val**2 + f**2)
        if map_name == "c1":
            # C1 map: f / (rho + |f|)
            denom = jnp.maximum(rho_val + jnp.abs(f), 1e-12)
            return f / denom

        raise ValueError(
            f"Invalid smooth_control_map: {map_name}. Expected one of: sqrt, c1."
        )

    def map_to_bounds(self, s, u_min, u_max):
        """
        Affine map from s in [-1, 1] to [u_min, u_max].
        """
        u_min_arr = jnp.asarray(u_min, dtype=s.dtype)
        u_max_arr = jnp.asarray(u_max, dtype=s.dtype)
        center = 0.5 * (u_max_arr + u_min_arr)
        half_span = 0.5 * (u_max_arr - u_min_arr)
        return center + half_span * s

    ### Model Output & Gradients ###

    def calc_u(self, params, x, t):
        """
        Calculate value function with hard constraint: V(x,t) = l(x) - phi(x,t)^2
        This ensures V(x,t) <= l(x) automatically.
        
        If terminal_hard_constraint is enabled, uses a normalized time factor
        so that V(x,T) = l(x) exactly at the configured terminal time T.
        """
        phi = self.model.apply(params, x, t)
        l_val = self.l(x)

        if self.config.terminal_hard_constraint or self.config.hard_constraint:
            match self.config.hard_constraint_type:
                case 'quadratic':
                    act = lambda x: x**2
                case 'softplus':
                    act = lambda x: jax.nn.softplus(x)
                case 'swish':
                    act = lambda x: jax.nn.swish(x)
                case 'elu':
                    act = lambda x: jax.nn.elu(x)
                case 'none':
                    act = lambda x: x
                case _:
                    raise ValueError(f"Invalid hard constraint type: {self.config.hard_constraint_type}")
        
        if self.config.terminal_hard_constraint:
            if self.config.alternative_tc:
                phi_T = self.model.apply(params, x, jnp.full_like(t, self.config.t_range[1]))
                phi = phi - phi_T
                return l_val - act(phi)
            else:
                # Terminal hard constraint: V(x,T) = l(x) at configured terminal time T
                t0, t1 = self.config.t_range
                horizon = jnp.maximum(t1 - t0, 1e-12)
                time_factor = (t1 - t) / horizon
                return l_val - time_factor * act(phi)
        
        elif self.config.hard_constraint:
            return l_val - act(phi)
        return phi
    
    def calc_ux(self,params,x,t,):
        def u_fn(x,t):
            u = self.calc_u(params,x,t)
            return u[...,(0,)]  # tuple index preserves (1,) shape so len(u)=1
        def jacrev(x,t):
            u,vjp_fun = jax.vjp(u_fn,x,t)
            ret = jax.vmap(vjp_fun,in_axes=0)(jnp.eye(len(u)))
            return u,ret[0],ret[1]
        return jax.vmap(jacrev,in_axes=0)(x,t)
    
    def calc_uxx(self,params,x,t):
        def jacfwd(x,t):
            def u_fn(x):
                u = self.calc_u(params,x,t)
                return u[...,0]
            def jacrev(x):
                u,vjp_fun = jax.vjp(u_fn,x)
                ret = jax.vmap(vjp_fun,in_axes=0)(jnp.eye(len(u)))
                return ret[0],u
            ux_fn = lambda s: jax.jvp(jacrev,(x,),(s,),has_aux=True)
            u_x,u_xx,u = jax.vmap(ux_fn,in_axes=1,out_axes=(None,1,None))(jnp.eye(len(x)))
            return u,u_x,u_xx
        return jax.vmap(jacfwd,in_axes=0)(x,t)

    ### Loss Helper Functions ###

    def sample_domain(self,key:Key,num_samples):
        x_pde = []
        for i in range(self.config.d_in):
            x_range = self.config.x_range[i] if i < len(self.config.x_range) else self.config.x_range[-1]
            x_pde.append(jax.random.uniform(key.newkey(),(num_samples,1),
                                            minval=x_range[0],maxval=x_range[1]))
        x_pde = jnp.hstack(x_pde)
        t_pde = jax.random.uniform(key.newkey(),(num_samples,1),
                                   minval=self.config.t_range[0],
                                   maxval=self.config.t_range[1])
        return x_pde,t_pde

    def sample_domain_eval(self, key: Key, num_samples):
        """
        Evaluation-time sampler.
        Defaults to training sampler for backward compatibility.
        """
        return self.sample_domain(key, num_samples)

    def sample_domain_t0(self,key:Key,num_samples):
        x,t = self.sample_domain(key,num_samples)
        t = jnp.zeros_like(t)
        return x,t

    def sample_domain_t0_eval(self, key: Key, num_samples):
        """
        Evaluation-time sampler with t fixed at 0.
        Defaults to sample_domain_eval for backward compatibility.
        """
        x, t = self.sample_domain_eval(key, num_samples)
        t = jnp.zeros_like(t)
        return x, t

    def supports_rad_sampling(self, loss_method: str) -> bool:
        """RAD currently targets VI-PINNs style PDE sampling losses."""
        return loss_method in ('vipinns', 'pinns')

    @partial(jax.jit, static_argnums=[0])
    def vipinns_vi_residual_abs(self, params, x, t):
        """
        Pointwise absolute VI residual used by RAD:
            |min(dV/dt + H, l - V)|
        """
        u, u_x, u_t = self.calc_ux(params, x, t)
        H = self.H(x, u_x)
        l = self.l(x)
        pde = u_t[..., 0] + H
        constraint = l - u
        vi = jnp.minimum(pde, constraint)
        return jnp.abs(vi[:, 0])

    def _compute_vi_residual_abs_chunked(self, params, x, t, chunk_size: int):
        """Chunked residual evaluation to bound memory on high-dimensional problems."""
        n = x.shape[0]
        chunk_size = max(1, int(chunk_size))
        residuals = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            residuals.append(self.vipinns_vi_residual_abs(params, x[start:end], t[start:end]))
        return jnp.concatenate(residuals, axis=0)

    def build_rad_candidate_pool(self, params, key: Key):
        """
        Build RAD candidate pool and sampling probabilities:
            1) Draw candidate pool from sample_domain
            2) Score VI residuals
            3) Build RAD probabilities over candidate points
        """
        candidate_size = max(int(self.config.batch_pde), int(self.config.rad_candidate_size))
        x_cand, t_cand = self.sample_domain(key, candidate_size)

        residual_abs = self._compute_vi_residual_abs_chunked(
            params,
            x_cand,
            t_cand,
            chunk_size=self.config.rad_residual_batch_size,
        )

        k = max(float(self.config.rad_k), 0.0)
        c = max(float(self.config.rad_c), 0.0)
        eps = max(float(self.config.rad_residual_eps), 1e-12)

        residual_pow = jnp.power(jnp.maximum(residual_abs, 0.0) + eps, k)
        mean_pow = jnp.mean(residual_pow)
        weights = residual_pow / (mean_pow + eps) + c
        weights = jnp.where(jnp.isfinite(weights), weights, 0.0)
        weights = jnp.maximum(weights, 0.0)

        weight_sum = float(jnp.sum(weights))
        if weight_sum <= 0.0:
            probs = jnp.ones((candidate_size,)) / candidate_size
        else:
            probs = weights / weight_sum

        return x_cand, t_cand, probs

    def sample_rad_batch_from_pool(self, rad_pool, key: Key):
        """
        Sample PDE points from a precomputed RAD candidate pool.
        """
        x_cand, t_cand, probs = rad_pool
        candidate_size = x_cand.shape[0]

        replace = int(self.config.batch_pde) > candidate_size
        idx = jax.random.choice(
            key.newkey(),
            candidate_size,
            (self.config.batch_pde,),
            replace=replace,
            p=probs,
        )
        return x_cand[idx], t_cand[idx]

    def sample_rad_pde_batch(self, params, key: Key):
        """
        Backward-compatible helper:
            Build a candidate pool and sample one RAD batch from it.
        """
        rad_pool = self.build_rad_candidate_pool(params, key)
        return self.sample_rad_batch_from_pool(rad_pool, key)
    
    def b(self,x,u_x = None, *, smooth: bool = False):
        """
        Drift for forward rollout under optimal control/disturbance.
        """
        return self.f(
            x,
            self.u_star(x, u_x, smooth=smooth),
            self.d_star(x, u_x, smooth=smooth),
        )
    
    def sigma(self,x):
        """
        Diffusion coefficient for forward SDE.
        For HJI, we use isotropic noise: sigma * I
        Note: In notebook, sigma is applied as scalar multiplication, not matrix.
        
        Args:
            x: State, shape (batch, d_in)
        
        Returns:
            Diffusion matrix, shape (batch, d_in, d_in)
        """
        noise = self.config.sigma_noise
        eye = noise * jnp.eye(x.shape[-1])
        return jnp.broadcast_to(eye[jnp.newaxis, :, :],
                                (x.shape[0], x.shape[-1], x.shape[-1]))
    
    def h(self,x,y,z,t):
        pass
    
    ### Losses ###
    
    def vipinns_loss(self,params,key:Key,x_pde=None,t_pde=None):
        if x_pde is None or t_pde is None:
            x_pde,t_pde = self.sample_domain(key,self.config.batch_pde)
        pde_loss =  self.vipinns_pde_loss(params,key,x_pde,t_pde)
        x_term,_ = self.sample_domain(key,self.config.batch_ic)
        term_loss = self.term_loss(params,key,x_term)
        return pde_loss+term_loss

    def vipinns_pde_loss(self,params,key:Key,x,t):
        u,u_x,u_t = self.calc_ux(params,x,t)
        H = self.H(x,u_x)
        l = self.l(x)
        pde = u_t[...,0] + H 
        constraint = l - u
        vi = jnp.minimum(pde,constraint)
        loss = jnp.mean(jnp.square(vi))
        return(loss*self.config.pde_scale,)
    
    def term_loss(self,params,key:Key,x):
        t = jnp.ones_like(x[:,0:1]) * self.config.t_range[1]
        if self.config.term_grad_loss:
            u,u_x,_ = self.calc_ux(params,x,t)
            l = self.l(x)
            loss = jnp.mean(jnp.square(l-u))
            lx_fn = jax.grad(lambda s: self.l(s)[..., 0])
            lx = jax.vmap(lx_fn)(x)
            loss += jnp.mean(jnp.square(lx-u_x[:,0,:]))
        else:
            u = self.calc_u(params,x,t)
            l = self.l(x)
            loss = jnp.mean(jnp.square(l-u))
        return (loss*self.config.ic_scale,)

    def fspinns_loss(self,params,key:Key):
        rollout_traj_count = self.config.batch_traj
        x_traj,_ = self.sample_domain_t0(key,rollout_traj_count)
        dt = jnp.zeros((rollout_traj_count, self.config.traj_len + 1, 1))
        dw = jnp.zeros((rollout_traj_count, self.config.traj_len + 1, self.config.d_in))
        dt = dt.at[:, 1:, :].set(self.config.delta_t)
        dw = dw.at[:, 1:, :].set(
            jnp.sqrt(self.config.delta_t)
            * jax.random.normal(
                key.newkey(),
                (rollout_traj_count, self.config.traj_len, self.config.d_in),
            )
        )
        t = jnp.cumsum(dt, axis=1)
        x = jnp.zeros((rollout_traj_count, self.config.traj_len + 1, self.config.d_in))
        x = x.at[:, 0, :].set(x_traj)
        smooth_controls = self._use_smooth_controls(training=True)

        def loop(i, input):
            x = input
            u, u_x, _ = self.calc_ux(params, x[:, i - 1, :], t[:, i - 1, :])
            x_next = (
                x[:, i - 1, :]
                + self.b(x[:, i - 1, :], u_x, smooth=smooth_controls) * self.config.delta_t
                + jnp.matmul(self.sigma(x[:, i - 1, :]), dw[:, i, :, jnp.newaxis])[..., 0]
            )
            x_next = self.project_state_to_domain(x_next)
            x = x.at[:, i, :].set(x_next)
            return x

        x = jax.lax.fori_loop(1, self.config.traj_len + 1, loop, (x))
        if self.config.stop_grad:
            x = jax.lax.stop_gradient(x)
        x_ic = x[:, -1, :]
        t_ic = t[:, -1, :]
        x = jnp.reshape(x, (-1, self.config.d_in))
        t = jnp.reshape(t, (-1, 1))
        temp = jnp.concatenate([x, t], axis=-1)
        n_candidates = temp.shape[0]
        if self.config.random_sample:
            temp = jax.random.choice(
                key.newkey(),
                temp,
                (self.config.batch_pde,),
                replace=self.config.batch_pde > n_candidates,
                axis=0,
            )
        elif self.config.batch_pde < n_candidates:
            temp = temp[: self.config.batch_pde]
        elif self.config.batch_pde > n_candidates:
            reps = (self.config.batch_pde + n_candidates - 1) // n_candidates
            temp = jnp.tile(temp, (reps, 1))[: self.config.batch_pde]
        x = temp[:, 0:-1]
        t = temp[:, -1:]

        pde_loss = self.vipinns_pde_loss(params, key, x, t)

        term_loss = self.term_loss(params,key,x_ic)

        return pde_loss + term_loss
    
    ### HJI-Specific Functions ###

    def f(self, x, u, d):
        """
        System dynamics: dx/dt = f(x, u, d)
        
        Args:
            x: State, shape (batch, d_in)
            u: Control input, shape (batch, d_control) or (batch, 1)
            d: Disturbance input, shape (batch, d_disturbance) or (batch, 1)
        
        Returns:
            dx/dt: shape (batch, d_in)
        """
        raise NotImplementedError("Dynamics f(x, u, d) must be implemented in subclass")
    
    def l(self, x):
        """
        Signed distance function / cost function.
        For reachability: l(x) < 0 indicates the target/constraint set.
        
        Args:
            x: State, shape (batch, d_in)
        
        Returns:
            Signed distance values, shape (batch, 1)
        """
        raise NotImplementedError("Signed distance l(x) must be implemented in subclass")

    def u_star(self, x, dv, *, smooth: bool = False):
        """
        Optimal control that MAXIMIZES the Hamiltonian (for reachability).
        u* = argmax_u { p · f(x, u, d) }
        
        Args:
            x: State, shape (batch, d_in)
            dv: Gradient of value function nabla_x V, shape (batch, d_out, d_in)
        
        Returns:
            Optimal control, shape (batch, 1) or (batch, d_control)
        """
        raise NotImplementedError("Optimal control u_star must be implemented in subclass")
    
    def d_star(self, x, dv, *, smooth: bool = False):
        """
        Optimal disturbance that MINIMIZES the Hamiltonian (for reachability).
        d* = argmin_d { p · f(x, u, d) }
        
        Args:
            x: State, shape (batch, d_in)
            dv: Gradient of value function nabla_x V, shape (batch, d_out, d_in)
        
        Returns:
            Optimal disturbance, shape (batch, 1) or (batch, d_disturbance)
        """
        raise NotImplementedError("Optimal disturbance d_star must be implemented in subclass")
    
    def H(self, x, dv):
        """
        Hamiltonian under optimal control and disturbance.
        H(x, p) = p · f(x, u*, d*)
        
        Args:
            x: State, shape (batch, d_in)
            dv: Gradient of value function nabla_x V, shape (batch, d_out, d_in)
        
        Returns:
            Hamiltonian values, shape (batch, 1)
        """
        smooth_controls = self._use_smooth_controls(training=False)
        u_opt = self.u_star(x, dv, smooth=smooth_controls)
        d_opt = self.d_star(x, dv, smooth=smooth_controls)
        dv_flat = dv[:, 0, :]  # (batch, d_in)
        f_val = self.f(x, u_opt, d_opt)
        return jax.vmap(jnp.inner, in_axes=(0, 0))(dv_flat, f_val)[..., jnp.newaxis]
    
    ### Safety Evaluation Functions ###
    
    def evaluate_safety_metric(self, params, key):
        """
        Evaluate safety metrics via the canonical ModelEvaluator implementation.
        """
        from evaluation import ModelEvaluator

        evaluator = ModelEvaluator(self, params=params, backend='jax')
        metrics = evaluator.compute_f1(
            N_rollouts=self.config.num_safety_rollouts,
            batch_size=self.config.safety_rollout_batch_size,
            key=key,
        )
        total = metrics['tp'] + metrics['fp'] + metrics['tn'] + metrics['fn']

        return {
            'safety_precision': float(metrics['precision']),
            'safety_recall': float(metrics['recall']),
            'safety_f1': float(metrics['f1']),
            'safety_accuracy': float(metrics['accuracy']),
            'safety_tp': int(metrics['tp']),
            'safety_fp': int(metrics['fp']),
            'safety_tn': int(metrics['tn']),
            'safety_fn': int(metrics['fn']),
            'safety_total_evaluated': int(total)
        }

    ### Misc ###
    
    @staticmethod
    def get_base_config(d_in=3, traj_len=50):
        """Get base configuration for HJI problems."""
        T = 1
        return Config(
            case='hji',
            d_in=d_in,
            d_out=1,
            traj_len=traj_len,
            delta_t=T / traj_len,
            x_range=((-1, 1),) * d_in,
            t_range=(0, T),
            random_sample=True,
        )
    
    ### Model Save/Load ###
    
    def save_model(self, params, run_name: str = None, save_dir: str = "./runs", 
                   extra_data: dict = None) -> str:
        """
        Save the trained model, config, and parameters to a directory.
        
        Creates a folder structure:
            ./runs/RUN_NAME/
                config.json       - Configuration as JSON
                params.pkl        - Model parameters (pickle)
                metadata.json     - Metadata (timestamp, solver class, etc.)
                extra_data.pkl    - Any additional data (optional)
        
        Args:
            params: Trained model parameters (JAX pytree)
            run_name: Name for the run folder. If None, generates timestamp-based name.
            save_dir: Base directory for runs (default: ./runs)
            extra_data: Optional dict of additional data to save
            
        Returns:
            run_path: Full path to the saved run directory
        """
        # Generate run name if not provided
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{self.config.case}_{timestamp}"
        
        # Create run directory
        run_path = os.path.join(save_dir, run_name)
        os.makedirs(run_path, exist_ok=True)
        
        # Save config as JSON
        config_dict = {}
        for key, value in vars(self.config).items():
            # Convert tuples to lists for JSON serialization
            if isinstance(value, tuple):
                config_dict[key] = list(value)
            else:
                config_dict[key] = value
        
        config_path = os.path.join(run_path, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save parameters as pickle
        params_path = os.path.join(run_path, "params.pkl")
        with open(params_path, 'wb') as f:
            pickle.dump(params, f)
        
        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'solver_class': self.__class__.__name__,
            'num_params': sum(x.size for x in jax.tree_util.tree_leaves(params)),
            'config_case': self.config.case,
            'd_in': self.config.d_in,
            'd_out': self.config.d_out,
        }
        metadata_path = os.path.join(run_path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save extra data if provided
        if extra_data is not None:
            extra_path = os.path.join(run_path, "extra_data.pkl")
            with open(extra_path, 'wb') as f:
                pickle.dump(extra_data, f)
        
        print(f"Model saved to: {run_path}")
        return run_path
    
    @staticmethod
    def load_model(run_path: str) -> dict:
        """
        Load a saved model from a run directory.
        
        Args:
            run_path: Path to the run directory
            
        Returns:
            Dictionary containing:
                'params': Model parameters
                'config': Config object
                'metadata': Metadata dict
                'extra_data': Extra data (if exists, else None)
        """
        # Load config
        config_path = os.path.join(run_path, "config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Convert lists back to tuples for specific fields
        tuple_fields = ['x_range', 't_range', 'save_layers', 'skip_layers', 'periodic_idx']
        for field in tuple_fields:
            if field in config_dict:
                value = config_dict[field]
                if isinstance(value, list):
                    # Handle nested lists (like x_range)
                    if len(value) > 0 and isinstance(value[0], list):
                        config_dict[field] = tuple(tuple(v) for v in value)
                    else:
                        config_dict[field] = tuple(value)
        
        # Create Config object
        config = Config(**{k: v for k, v in config_dict.items() if hasattr(Config, k)})
        
        # Load parameters
        params_path = os.path.join(run_path, "params.pkl")
        with open(params_path, 'rb') as f:
            params = pickle.load(f)
        
        # Load metadata
        metadata_path = os.path.join(run_path, "metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load extra data if exists
        extra_data = None
        extra_path = os.path.join(run_path, "extra_data.pkl")
        if os.path.exists(extra_path):
            with open(extra_path, 'rb') as f:
                extra_data = pickle.load(f)
        
        return {
            'params': params,
            'config': config,
            'metadata': metadata,
            'extra_data': extra_data
        }
    
    ### optimization method ###
    @partial(jax.jit, static_argnums=[0,1])
    def optimize(self,loss_method,key:Key,params,opt_state):
        def loss_fn(key,params):
            losses = jnp.asarray(self.select_loss(loss_method)(params,key))
            loss = jnp.sum(losses)
            return loss, losses
        (loss,losses), grad = jax.value_and_grad(loss_fn,argnums=1,has_aux=True)(key,params)
        updates, opt_state = self.optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

        return loss,losses,params,opt_state,key

    @partial(jax.jit, static_argnums=[0,1])
    def optimize_with_pde_batch(self, loss_method, key: Key, params, opt_state, x_pde, t_pde):
        """
        Optimization step using externally provided PDE collocation points.
        Used for RAD-enabled training loops.
        """
        def loss_fn(key, params):
            match loss_method:
                case 'vipinns' | 'pinns':
                    losses = jnp.asarray(self.vipinns_loss(params, key, x_pde, t_pde))
                case _:
                    losses = jnp.asarray(self.select_loss(loss_method)(params, key))
            loss = jnp.sum(losses)
            return loss, losses

        (loss, losses), grad = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)(key, params)
        updates, opt_state = self.optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

        return loss, losses, params, opt_state, key


class HJI_Controller():
    def __init__(self,solver:HJI_Solver,seed = 1234):
        self.solver = solver
        self.key = Key.create_key(seed)
        self.params,self.opt_state = self.solver.init_solver(self.key)
        self.track = []
        self.training_configs = []
        self.rad_candidate_pool = None
        self._rad_warning_shown = False

    def step(self,loss_method,i):
        self.key.change()

        rad_active = (
            self.solver.config.use_rad_sampling
            and i >= self.solver.config.rad_warmup_iters
            and self.solver.supports_rad_sampling(loss_method)
        )

        if (
            self.solver.config.use_rad_sampling
            and not self.solver.supports_rad_sampling(loss_method)
            and not self._rad_warning_shown
        ):
            print(f"RAD sampling is enabled but not supported for loss '{loss_method}'. Falling back to default sampling.")
            self._rad_warning_shown = True

        if rad_active:
            refresh_interval = max(1, int(self.solver.config.rad_refresh_interval))
            needs_refresh = (self.rad_candidate_pool is None) or (i % refresh_interval == 0)
            if needs_refresh:
                self.rad_candidate_pool = self.solver.build_rad_candidate_pool(self.params, self.key)
            x_pde, t_pde = self.solver.sample_rad_batch_from_pool(self.rad_candidate_pool, self.key)
            loss,losses,self.params,self.opt_state,self.key = self.solver.optimize_with_pde_batch(
                loss_method, self.key, self.params, self.opt_state, x_pde, t_pde
            )
        else:
            self.rad_candidate_pool = None
            loss,losses,self.params,self.opt_state,self.key = self.solver.optimize(
                loss_method,self.key,self.params,self.opt_state
            )
        self.track.append(loss)

        # Saving logs
        if self.solver.config.save_to_wandb:
            temp = {"loss"+str(k+1):v for k,v in dict(enumerate(losses)).items()}
            wandb.log(temp,commit=False)
            wandb.log({"loss": loss})
    
    def solve(
        self,
        checkpoint_callback: Optional[Callable] = None,
        checkpoint_steps: Optional[Set[int]] = None,
        include_stage_end_checkpoints: bool = False
    ):
        checkpoint_steps = set() if checkpoint_steps is None else {int(s) for s in checkpoint_steps}
        if any(s < 0 for s in checkpoint_steps):
            raise ValueError("checkpoint_steps must contain non-negative integers.")

        callback_enabled = callable(checkpoint_callback)
        # Preserve exact legacy training behavior when checkpointing is not requested.
        if (not callback_enabled) and (not include_stage_end_checkpoints) and (len(checkpoint_steps) == 0):
            for i in tqdm.tqdm(range(self.solver.config.iter)):
                self.step(self.solver.config.loss_method,i)
            if self.solver.config.additional_losses:
                loss_num=1
                for config in self.training_configs:
                    self.change_config(config,loss_num)
                    for i in tqdm.tqdm(range(self.solver.config.iter)):
                        self.step(self.solver.config.loss_method,i)
                    loss_num+=1
            if self.solver.config.safety_eval and hasattr(self.solver,'evaluate_safety_metric'):
                print("\nComputing safety evaluation metric...")
                safety_metrics = self.solver.evaluate_safety_metric(self.params, self.key)
                print(f"Safety Metrics:")
                print(f"  Precision: {safety_metrics['safety_precision']:.4f}")
                print(f"  Recall:    {safety_metrics['safety_recall']:.4f}")
                print(f"  F1 Score:  {safety_metrics['safety_f1']:.4f}")
                print(f"  Accuracy:  {safety_metrics['safety_accuracy']:.4f}")
                print(f"  TP: {safety_metrics['safety_tp']}, FP: {safety_metrics['safety_fp']}, "
                      f"TN: {safety_metrics['safety_tn']}, FN: {safety_metrics['safety_fn']}")
                print(f"  Total evaluated: {safety_metrics['safety_total_evaluated']}")
                if self.solver.config.save_to_wandb:
                    wandb.summary.update(safety_metrics)
            if self.solver.config.auto_close:
                self.solver.close()
            return

        callback_seen_steps = set()

        def maybe_emit_checkpoint(
            global_step: int,
            stage_idx: int,
            stage_iter_idx: int,
            stage_total_iters: int,
            trigger: str
        ):
            if not callback_enabled:
                return

            emit = False
            if global_step in checkpoint_steps:
                emit = True
            if include_stage_end_checkpoints and stage_total_iters > 0 and stage_iter_idx == stage_total_iters:
                emit = True
            if not emit:
                return
            if global_step in callback_seen_steps:
                return

            callback_meta = {
                "global_step": int(global_step),
                "stage_idx": int(stage_idx),
                "stage_iter_idx": int(stage_iter_idx),
                "stage_total_iters": int(stage_total_iters),
                "loss_method": str(self.solver.config.loss_method),
                "trigger": str(trigger),
            }
            checkpoint_callback(self.params, callback_meta)
            callback_seen_steps.add(global_step)

        def run_stage(stage_idx: int, stage_total_iters: int, global_step: int):
            if stage_total_iters <= 0:
                return global_step

            for iter_idx in tqdm.tqdm(range(stage_total_iters)):
                self.step(self.solver.config.loss_method, iter_idx)
                global_step += 1
                maybe_emit_checkpoint(
                    global_step=global_step,
                    stage_idx=stage_idx,
                    stage_iter_idx=iter_idx + 1,
                    stage_total_iters=stage_total_iters,
                    trigger="post_step",
                )
            return global_step

        global_step = 0
        stage_idx = 0
        maybe_emit_checkpoint(
            global_step=global_step,
            stage_idx=stage_idx,
            stage_iter_idx=0,
            stage_total_iters=int(self.solver.config.iter),
            trigger="pre_training",
        )

        global_step = run_stage(
            stage_idx=stage_idx,
            stage_total_iters=int(self.solver.config.iter),
            global_step=global_step,
        )

        if self.solver.config.additional_losses:
            loss_num = 1
            for config in self.training_configs:
                stage_idx += 1
                self.change_config(config, loss_num)
                global_step = run_stage(
                    stage_idx=stage_idx,
                    stage_total_iters=int(self.solver.config.iter),
                    global_step=global_step,
                )
                loss_num += 1
        if self.solver.config.safety_eval and hasattr(self.solver,'evaluate_safety_metric'):
            print("\nComputing safety evaluation metric...")
            safety_metrics = self.solver.evaluate_safety_metric(self.params, self.key)
            print(f"Safety Metrics:")
            print(f"  Precision: {safety_metrics['safety_precision']:.4f}")
            print(f"  Recall:    {safety_metrics['safety_recall']:.4f}")
            print(f"  F1 Score:  {safety_metrics['safety_f1']:.4f}")
            print(f"  Accuracy:  {safety_metrics['safety_accuracy']:.4f}")
            print(f"  TP: {safety_metrics['safety_tp']}, FP: {safety_metrics['safety_fp']}, "
                  f"TN: {safety_metrics['safety_tn']}, FN: {safety_metrics['safety_fn']}")
            print(f"  Total evaluated: {safety_metrics['safety_total_evaluated']}")
            if self.solver.config.save_to_wandb:
                wandb.summary.update(safety_metrics)
        if self.solver.config.auto_close:
            self.solver.close()
    
    def close(self):
        self.solver.close()
    
    def append_train_config(self,config:TrainConfig):
        self.training_configs.append(config)

    def change_config(self,config:TrainConfig,loss_num):
        for var in vars(config):
            if getattr(self.solver.config, var, None) != vars(config)[var]:
                setattr(self.solver.config,var,vars(config)[var])
                if self.solver.config.save_to_wandb:
                    wandb.config[var+str(loss_num)] = vars(config)[var]

        self.opt_state.hyperparams['learning_rate'] = self.solver.config.lr
