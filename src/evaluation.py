"""
Model Evaluation Module for HJI Value Function Networks.

Provides evaluation metrics for trained HJI value function models including:
- F1 statistics (precision, recall, F1 score) via trajectory rollouts
- Predicted safe set volume estimation (N_safe / N_total percentage)
- PDE residual error (VI residual from vipinns_loss)

All metrics use the zero-level set threshold (V(x,0) > 0) for consistency
with hji_solver.evaluate_safety_metric.

Supports both JAX/Flax models and PyTorch JIT-compiled models.

Usage:
    # For JAX model
    evaluator = ModelEvaluator(solver, params=params)
    result = evaluator.evaluate(N_rollouts=10000, N_residual_samples=10000)
    
    # For Torch model
    evaluator = ModelEvaluator(solver, torch_model_path="model.pt")
    result = evaluator.evaluate(N_rollouts=10000, N_residual_samples=10000)
"""

import sys
sys.path.append('..')
sys.path.append('.')
import warnings
from pathlib import Path
import jax
import jax.numpy as jnp
from jax import vmap
from functools import partial
import numpy as np
from typing import Dict, Tuple, Optional, Any, Union, Callable, Sequence, List
from dataclasses import dataclass

# Try to import torch for optional support
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class ModelEvalResult:
    """
    Results from model evaluation.
    
    All metrics use threshold delta=0 (zero-level set).
    
    Attributes:
        # F1 statistics
        precision: TP / (TP + FP) - fraction of predicted safe that are actually safe
        recall: TP / (TP + FN) - fraction of actually safe that are predicted safe
        f1: 2 * precision * recall / (precision + recall)
        accuracy: (TP + TN) / total
        true_volume: (TP + FN) / total - fraction of actually safe points
        tp: True positives (predicted safe AND actually safe)
        fp: False positives (predicted safe BUT actually unsafe)
        tn: True negatives (predicted unsafe AND actually unsafe)
        fn: False negatives (predicted unsafe BUT actually safe)
        
        # Volume estimation
        volume_percentage: N_safe / N_total * 100 (percentage of domain predicted safe)
        N_safe: Number of samples where V(x, 0) > 0
        N_total_volume: Total number of samples used for volume estimation
        
        # PDE residual
        pde_residual_mean: Mean of VI residual squared
        pde_residual_max: Maximum VI residual squared
        pde_residual_std: Standard deviation of VI residual
        
        # Evaluation parameters
        N_rollouts: Number of trajectories rolled out for F1 computation
        N_residual_samples: Number of samples used for PDE residual computation
        backend: 'jax' or 'torch'

        # Ground truth comparison
        ground_truth_mse_t0: Optional mean squared error at t=0.
        ground_truth_rmse_t0: Optional root mean squared error at t=0.
        ground_truth_rl2_t0: Optional relative L2 error at t=0 over the full
            available x-domain:
                RL2 = sqrt(sum((u_ref - u_pred)^2) / sum(u_ref^2))
        ground_truth_num_points: Number of domain points used for t=0 metrics
        ground_truth_artifact: Path to the artifact used for t=0 comparison
    """
    # F1 statistics
    precision: float
    recall: float
    f1: float
    accuracy: float
    true_volume: float
    tp: int
    fp: int
    tn: int
    fn: int
    
    # Volume estimation
    volume_percentage: float
    N_safe: int
    N_total_volume: int
    
    # PDE residual
    pde_residual_mean: float
    pde_residual_max: float
    pde_residual_std: float
    
    # Evaluation parameters
    N_rollouts: int
    N_residual_samples: int
    backend: str

    # Ground truth comparison
    ground_truth_mse_t0: Optional[float] = None
    ground_truth_rmse_t0: Optional[float] = None
    ground_truth_rl2_t0: Optional[float] = None
    ground_truth_num_points: Optional[int] = None
    ground_truth_artifact: Optional[str] = None
    
    def __repr__(self):
        return (f"ModelEvalResult(\n"
                f"  F1 Statistics:\n"
                f"    precision={self.precision:.4f}, recall={self.recall:.4f}, f1={self.f1:.4f}\n"
                f"    accuracy={self.accuracy:.4f}, true_volume={self.true_volume:.4f}\n"
                f"    confusion: TP={self.tp}, FP={self.fp}, TN={self.tn}, FN={self.fn}\n"
                f"  Volume:\n"
                f"    volume_percentage={self.volume_percentage:.2f}% ({self.N_safe}/{self.N_total_volume})\n"
                f"  PDE Residual:\n"
                f"    mean={self.pde_residual_mean:.6e}, max={self.pde_residual_max:.6e}, std={self.pde_residual_std:.6e}\n"
                f"  Ground Truth:\n"
                f"    mse_t0={self.ground_truth_mse_t0}, rmse_t0={self.ground_truth_rmse_t0}, rl2_t0={self.ground_truth_rl2_t0}\n"
                f"    points={self.ground_truth_num_points}\n"
                f"    artifact={self.ground_truth_artifact}\n"
                f"  Parameters:\n"
                f"    N_rollouts={self.N_rollouts}, N_residual={self.N_residual_samples}\n"
                f"    backend={self.backend}\n"
                f")")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for logging."""
        return {
            'eval_precision': self.precision,
            'eval_recall': self.recall,
            'eval_f1': self.f1,
            'eval_accuracy': self.accuracy,
            'eval_true_volume': self.true_volume,
            'eval_tp': self.tp,
            'eval_fp': self.fp,
            'eval_tn': self.tn,
            'eval_fn': self.fn,
            'eval_volume_percentage': self.volume_percentage,
            'eval_N_safe': self.N_safe,
            'eval_N_total_volume': self.N_total_volume,
            'eval_pde_residual_mean': self.pde_residual_mean,
            'eval_pde_residual_max': self.pde_residual_max,
            'eval_pde_residual_std': self.pde_residual_std,
            'eval_ground_truth_mse_t0': self.ground_truth_mse_t0,
            'eval_ground_truth_rmse_t0': self.ground_truth_rmse_t0,
            'eval_ground_truth_rl2_t0': self.ground_truth_rl2_t0,
            'eval_ground_truth_num_points': self.ground_truth_num_points,
            'eval_ground_truth_artifact': self.ground_truth_artifact,
            'eval_N_rollouts': self.N_rollouts,
            'eval_N_residual_samples': self.N_residual_samples,
            'eval_backend': self.backend,
        }


class ModelEvaluator:
    """
    Evaluator for trained HJI value function models.
    
    Supports both JAX/Flax models (with params dict) and PyTorch JIT-compiled models.
    Provides F1 statistics, volume estimation, and PDE residual computation.
    
    Usage:
        # For JAX model (using solver's model)
        evaluator = ModelEvaluator(solver, params=trained_params)
        
        # For Torch JIT model
        evaluator = ModelEvaluator(solver, torch_model="path/to/model.pt")
        # or
        evaluator = ModelEvaluator(solver, torch_model=torch.jit.load("model.pt"))
        
        # Load from saved run directory
        evaluator = ModelEvaluator.from_saved_model("./runs/my_run", SolverClass)
        
        result = evaluator.evaluate(N_rollouts=10000, N_residual_samples=10000)
    """
    
    def __init__(
        self,
        solver,
        params: Optional[Any] = None,
        torch_model: Optional[Union[str, Any]] = None,
        seed: int = 42,
        torch_state_permutation: Optional[Sequence[int]] = None,
        torch_state_center: Optional[Sequence[float]] = None,
        torch_state_scale: Optional[Sequence[float]] = None,
        torch_time_shift: float = 0.0,
        torch_time_scale: float = 1.0,
        torch_time_first: bool = True,
        torch_output_postprocess: Optional[Callable[[Any, Any, Any], Any]] = None,
    ):
        """
        Initialize evaluator with HJI solver and model.
        
        Args:
            solver: HJI_Solver instance (provides dynamics, l, sample_domain, etc.)
            params: JAX model parameters (if using JAX backend)
            torch_model: Path to TorchScript model or loaded torch.jit module
                         Model expects input [t, x] with t as first element
            seed: Random seed for reproducibility
            torch_state_permutation: Optional state index permutation applied before
                model input. Example [1, 0] maps solver state [z, v] -> model [v, z].
            torch_state_center: Optional per-dimension centering for model inputs.
            torch_state_scale: Optional per-dimension scaling divisor for model inputs.
            torch_time_shift: Optional shift applied to time before model input.
            torch_time_scale: Optional time scaling divisor before model input.
            torch_time_first: If True, model input is [t, x]; else [x, t].
            torch_output_postprocess: Optional callable applied to raw torch model
                output as: postprocess(y_raw, x_real, t_model).
                Use torch ops inside this function if gradient support is needed.
        """
        self.solver = solver
        self.config = solver.config
        self.key = jax.random.key(seed)
        
        # Rollout parameters
        self.dt = getattr(self.config, 'delta_t', 0.02)
        self.traj_len = getattr(self.config, 'traj_len', 50)
        self.d_in = self.config.d_in

        # Optional Torch preprocessing parameters.
        self._torch_state_permutation = None if torch_state_permutation is None else tuple(int(i) for i in torch_state_permutation)
        if self._torch_state_permutation is not None:
            if len(self._torch_state_permutation) != self.d_in:
                raise ValueError(
                    f"torch_state_permutation must have length d_in={self.d_in}, "
                    f"got {len(self._torch_state_permutation)}"
                )
            if sorted(self._torch_state_permutation) != list(range(self.d_in)):
                raise ValueError("torch_state_permutation must be a valid permutation of [0, ..., d_in-1].")

        self._torch_state_center_np = None if torch_state_center is None else np.asarray(torch_state_center, dtype=np.float32)
        self._torch_state_scale_np = None if torch_state_scale is None else np.asarray(torch_state_scale, dtype=np.float32)
        if self._torch_state_center_np is not None and self._torch_state_center_np.shape != (self.d_in,):
            raise ValueError(
                f"torch_state_center must have shape ({self.d_in},), got {self._torch_state_center_np.shape}"
            )
        if self._torch_state_scale_np is not None and self._torch_state_scale_np.shape != (self.d_in,):
            raise ValueError(
                f"torch_state_scale must have shape ({self.d_in},), got {self._torch_state_scale_np.shape}"
            )
        if self._torch_state_scale_np is not None and np.any(np.abs(self._torch_state_scale_np) < 1e-12):
            raise ValueError("torch_state_scale contains values too close to zero.")

        self._torch_time_shift = float(torch_time_shift)
        self._torch_time_scale = float(torch_time_scale)
        if abs(self._torch_time_scale) < 1e-12:
            raise ValueError("torch_time_scale must be non-zero.")
        self._torch_time_first = bool(torch_time_first)
        if torch_output_postprocess is not None and not callable(torch_output_postprocess):
            raise ValueError("torch_output_postprocess must be callable when provided.")
        self._torch_output_postprocess = torch_output_postprocess
        
        # Determine backend and set up prediction function
        if params is not None and torch_model is not None:
            raise ValueError("Cannot specify both params (JAX) and torch_model. Choose one backend.")
        
        if params is not None:
            self.backend = 'jax'
            self.params = params
            self._setup_jax_backend()
        elif torch_model is not None:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is required for torch_model support. Install with: pip install torch")
            self.backend = 'torch'
            self._setup_torch_backend(torch_model)
        else:
            raise ValueError("Must provide either params (JAX) or torch_model (Torch)")
    
    @classmethod
    def from_saved_model(
        cls,
        run_path: str,
        solver_class,
        seed: int = 42
    ) -> 'ModelEvaluator':
        """
        Create evaluator from a saved model directory.
        
        Args:
            run_path: Path to the saved run directory (e.g., "./runs/my_run")
            solver_class: The solver class to instantiate (e.g., VerticalDrone_HJI)
            seed: Random seed for reproducibility
            
        Returns:
            ModelEvaluator instance with loaded model
            
        Example:
            from problems import VerticalDrone_HJI
            evaluator = ModelEvaluator.from_saved_model("./runs/my_run", VerticalDrone_HJI)
            result = evaluator.evaluate()
        """
        from hji_solver import HJI_Solver
        
        # Load the saved model data
        loaded = HJI_Solver.load_model(run_path)
        params = loaded['params']
        config = loaded['config']
        
        # Create solver instance with loaded config
        solver = solver_class(config)
        
        return cls(solver, params=params, seed=seed)
    
    def _setup_jax_backend(self):
        """Setup JAX backend with JIT-compiled functions."""
        self._predict_fn = self._create_jax_predict_fn()
        self._predict_with_grad_fn = self._create_jax_predict_with_grad_fn()
    
    def _create_jax_predict_fn(self) -> Callable:
        """Create JAX prediction function V(x, t)."""
        def predict(x, t):
            """
            Predict value function.
            
            Args:
                x: State, shape (batch, d_in)
                t: Time, shape (batch, 1)
            
            Returns:
                V(x, t), shape (batch, 1)
            """
            return self.solver.calc_u(self.params, x, t)
        
        return jax.jit(predict)
    
    def _create_jax_predict_with_grad_fn(self) -> Callable:
        """Create JAX prediction function with gradients."""
        def predict_with_grad(x, t):
            """
            Predict value function with gradients.
            
            Args:
                x: State, shape (batch, d_in)
                t: Time, shape (batch, 1)
            
            Returns:
                V(x, t): shape (batch, 1)
                dV/dx: shape (batch, 1, d_in)
                dV/dt: shape (batch, 1, 1)
            """
            return self.solver.calc_ux(self.params, x, t)
        
        return jax.jit(predict_with_grad)
    
    def _setup_torch_backend(self, torch_model):
        """Setup PyTorch backend."""
        if isinstance(torch_model, str):
            self.torch_model = torch.jit.load(torch_model)
        else:
            self.torch_model = torch_model
        
        self.torch_model.eval()
        
        # Determine device
        try:
            # Try to get device from model parameters
            self.device = next(self.torch_model.parameters()).device
        except StopIteration:
            self.device = torch.device('cpu')

        # Cache preprocessing tensors on model device.
        self._torch_perm_index = None
        if self._torch_state_permutation is not None:
            self._torch_perm_index = torch.tensor(self._torch_state_permutation, dtype=torch.long, device=self.device)
        self._torch_state_center = None
        if self._torch_state_center_np is not None:
            self._torch_state_center = torch.tensor(self._torch_state_center_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        self._torch_state_scale = None
        if self._torch_state_scale_np is not None:
            self._torch_state_scale = torch.tensor(self._torch_state_scale_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Detect TorchScript input/output interface so we can support both:
        #   1) tensor input -> tensor output
        #   2) dict input   -> dict/tensor output (DeepReach-style)
        self._torch_input_adapter = None
        self._torch_input_mode = None
        self._infer_torch_model_interface()
        # Some scripted models detach inside forward(), which breaks gradients
        # w.r.t. external inputs. If available, prefer a direct `net(...)` call.
        self._torch_use_direct_net_for_grad = False
        self._infer_torch_direct_net_grad_path()
        
        self._predict_fn = self._create_torch_predict_fn()
        self._predict_with_grad_fn = self._create_torch_predict_with_grad_fn()

    def _infer_torch_model_interface(self):
        """Infer accepted Torch model input contract and validate output extraction."""
        sample_tx = torch.zeros((1, 1 + self.d_in), dtype=torch.float32, device=self.device)

        candidates = (
            ("tensor", lambda tx: tx),
            ("coords_dict", lambda tx: {"coords": tx}),
            ("model_input_dict", lambda tx: {"model_input": tx}),
            ("input_dict", lambda tx: {"input": tx}),
            ("x_dict", lambda tx: {"x": tx}),
        )

        last_exc = None
        for mode_name, adapter in candidates:
            try:
                with torch.no_grad():
                    output = self.torch_model(adapter(sample_tx))
                self._extract_torch_output_tensor(output)
                self._torch_input_mode = mode_name
                self._torch_input_adapter = adapter
                return
            except Exception as exc:
                last_exc = exc

        raise RuntimeError(
            "Unsupported Torch model interface. Expected tensor input or dict input "
            "with one of keys: coords, model_input, input, x."
        ) from last_exc

    def _extract_torch_output_tensor(self, output):
        """Normalize model output object to a tensor value prediction."""
        if torch.is_tensor(output):
            return output

        if isinstance(output, dict):
            preferred_keys = ("model_out", "output", "out", "value", "values", "y")
            for key in preferred_keys:
                value = output.get(key)
                if torch.is_tensor(value):
                    return value

            tensor_values = [value for value in output.values() if torch.is_tensor(value)]
            if len(tensor_values) == 1:
                return tensor_values[0]

            keys = list(output.keys())
            raise TypeError(
                f"Could not identify tensor output in dict. Available keys: {keys}"
            )

        if isinstance(output, (tuple, list)):
            for value in output:
                if torch.is_tensor(value):
                    return value
            raise TypeError("Model returned tuple/list without tensor outputs.")

        raise TypeError(f"Unsupported Torch model output type: {type(output)}")

    def _call_torch_model(self, tx_input):
        """Call Torch model with detected input adapter and return a (batch, 1)-like tensor."""
        if self._torch_input_adapter is None:
            raise RuntimeError("Torch model interface adapter was not initialized.")

        model_input = self._torch_input_adapter(tx_input)
        raw_output = self.torch_model(model_input)
        v_torch = self._extract_torch_output_tensor(raw_output)

        if v_torch.ndim == 0:
            v_torch = v_torch.reshape(1, 1)
        elif v_torch.ndim == 1:
            v_torch = v_torch.unsqueeze(-1)

        return v_torch

    def _infer_torch_direct_net_grad_path(self):
        """
        Detect whether `torch_model.net(tx)` is usable for gradient computation.

        DeepReach exported TorchScript models can detach inputs inside forward(),
        yielding zero/None gradients w.r.t. caller-provided tensors. Calling the
        underlying scripted submodule `net(...)` bypasses that detach path.
        """
        net = getattr(self.torch_model, "net", None)
        if net is None:
            return

        sample_tx = torch.zeros(
            (2, 1 + self.d_in),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        try:
            y_direct = net(sample_tx)
            y_direct = self._extract_torch_output_tensor(y_direct)
            if y_direct.ndim == 0:
                y_direct = y_direct.reshape(1, 1)
            elif y_direct.ndim == 1:
                y_direct = y_direct.unsqueeze(-1)

            g_direct = torch.autograd.grad(
                outputs=y_direct.sum(),
                inputs=sample_tx,
                allow_unused=True,
                create_graph=False,
                retain_graph=False,
            )[0]
            if g_direct is None:
                return

            # Ensure direct-net output numerically matches normal forward output.
            with torch.no_grad():
                y_fwd = self._call_torch_model(sample_tx.detach())
            if y_fwd.shape != y_direct.shape:
                return
            if not torch.allclose(y_fwd, y_direct.detach(), atol=1e-6, rtol=1e-5):
                return

            self._torch_use_direct_net_for_grad = True
        except Exception:
            self._torch_use_direct_net_for_grad = False

    def _call_torch_model_for_grad(self, tx_input):
        """
        Call torch model for gradient-enabled evaluation.

        Prefers direct `net(...)` path when available to preserve gradients.
        """
        if self._torch_use_direct_net_for_grad:
            raw_output = self.torch_model.net(tx_input)
            v_torch = self._extract_torch_output_tensor(raw_output)
            if v_torch.ndim == 0:
                v_torch = v_torch.reshape(1, 1)
            elif v_torch.ndim == 1:
                v_torch = v_torch.unsqueeze(-1)
            return v_torch
        return self._call_torch_model(tx_input)

    def _build_torch_model_input(self, x_torch, t_torch):
        """Build model input tensor with optional permutation/normalization."""
        x_model = x_torch
        if self._torch_perm_index is not None:
            x_model = torch.index_select(x_model, dim=-1, index=self._torch_perm_index)

        if self._torch_state_center is not None:
            x_model = x_model - self._torch_state_center
        if self._torch_state_scale is not None:
            x_model = x_model / self._torch_state_scale

        t_model = (t_torch - self._torch_time_shift) / self._torch_time_scale
        if self._torch_time_first:
            tx_input = torch.cat([t_model, x_model], dim=-1)
        else:
            tx_input = torch.cat([x_model, t_model], dim=-1)
        return tx_input, x_model, t_model

    def _apply_torch_output_postprocess(self, y_raw, x_real, t_model):
        """Apply optional output post-processing to raw torch predictions."""
        if self._torch_output_postprocess is None:
            return y_raw

        y_proc = self._torch_output_postprocess(y_raw, x_real, t_model)
        if not torch.is_tensor(y_proc):
            y_proc = torch.as_tensor(y_proc, dtype=y_raw.dtype, device=y_raw.device)

        if y_proc.ndim == 0:
            y_proc = y_proc.reshape(1, 1)
        elif y_proc.ndim == 1:
            y_proc = y_proc.unsqueeze(-1)
        return y_proc
    
    def _create_torch_predict_fn(self) -> Callable:
        """Create Torch prediction function V(x, t)."""
        def predict(x, t):
            """
            Predict value function using Torch model.
            
            Args:
                x: State, shape (batch, d_in) - JAX array or numpy
                t: Time, shape (batch, 1) - JAX array or numpy
            
            Returns:
                V(x, t), shape (batch, 1) - as numpy array
            """
            x_np = np.asarray(x)
            t_np = np.asarray(t)
            
            x_torch = torch.tensor(x_np, dtype=torch.float32, device=self.device)
            t_torch = torch.tensor(t_np, dtype=torch.float32, device=self.device)
            
            tx_input, _, t_model = self._build_torch_model_input(x_torch, t_torch)
            
            with torch.no_grad():
                v_torch = self._call_torch_model(tx_input)
                v_torch = self._apply_torch_output_postprocess(v_torch, x_torch, t_model)
            
            return np.array(v_torch.cpu())
        
        return predict
    
    def _create_torch_predict_with_grad_fn(self) -> Callable:
        """Create Torch prediction function with gradients."""
        def predict_with_grad(x, t):
            """
            Predict value function with gradients using Torch autograd.
            
            Args:
                x: State, shape (batch, d_in) - JAX array or numpy
                t: Time, shape (batch, 1) - JAX array or numpy
            
            Returns:
                V(x, t): shape (batch, 1) - as numpy array
                dV/dx: shape (batch, 1, d_in) - as numpy array
                dV/dt: shape (batch, 1, 1) - as numpy array
            """
            x_np = np.asarray(x)
            t_np = np.asarray(t)
            
            x_torch = torch.tensor(x_np, dtype=torch.float32, device=self.device, requires_grad=True)
            t_torch = torch.tensor(t_np, dtype=torch.float32, device=self.device, requires_grad=True)
            
            tx_input, _, t_model = self._build_torch_model_input(x_torch, t_torch)
            
            v_torch = self._call_torch_model_for_grad(tx_input)
            v_torch = self._apply_torch_output_postprocess(v_torch, x_torch, t_model)
            
            # Compute gradients
            grad_outputs = torch.ones_like(v_torch)
            
            grads = torch.autograd.grad(
                outputs=v_torch,
                inputs=[x_torch, t_torch],
                grad_outputs=grad_outputs,
                allow_unused=True,
                create_graph=False,
                retain_graph=False
            )
            
            dv_dx = grads[0]
            dv_dt = grads[1]

            # Some scripted models may ignore part of the input (often t),
            # which makes autograd return None when allow_unused=True.
            if dv_dx is None:
                dv_dx = torch.zeros_like(x_torch)
            if dv_dt is None:
                dv_dt = torch.zeros_like(t_torch)
            
            # Reshape to match JAX convention: dV/dx has shape (batch, 1, d_in)
            v_np = v_torch.detach().cpu().numpy()
            dv_dx_np = dv_dx.detach().cpu().numpy()[:, np.newaxis, :]  # (batch, 1, d_in)
            dv_dt_np = dv_dt.detach().cpu().numpy()[:, np.newaxis, :]  # (batch, 1, 1)
            
            return v_np, dv_dx_np, dv_dt_np
        
        return predict_with_grad
    
    def _split_key(self):
        """Get a new random key."""
        self.key, subkey = jax.random.split(self.key)
        return subkey

    def _is_pubsub_case(self) -> bool:
        """Return True when evaluator is running a PubSub reachability case."""
        case = getattr(self.solver.config, 'case', None)
        return case in ('pubsub_nd', 'pubsub_40d')
    
    def predict(self, x, t) -> np.ndarray:
        """
        Predict value function V(x, t).
        
        Args:
            x: State, shape (batch, d_in)
            t: Time, shape (batch, 1)
        
        Returns:
            V(x, t), shape (batch, 1)
        """
        result = self._predict_fn(x, t)
        return np.asarray(result)
    
    def predict_with_grad(self, x, t) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict value function with gradients.
        
        Args:
            x: State, shape (batch, d_in)
            t: Time, shape (batch, 1)
        
        Returns:
            V(x, t): shape (batch, 1)
            dV/dx: shape (batch, 1, d_in)
            dV/dt: shape (batch, 1, 1)
        """
        v, dv_dx, dv_dt = self._predict_with_grad_fn(x, t)
        return np.asarray(v), np.asarray(dv_dx), np.asarray(dv_dt)
    
    def compute_pde_residual(
        self,
        N_samples: int = 10000,
        batch_size: int = 1000,
        key=None
    ) -> Dict[str, float]:
        """
        Compute PDE residual (VI residual) over sampled domain points at t=0.
        
        The VI residual is: r(x, 0) = min(dV/dt + H(x, ∇V), l(x) - V(x, 0))
        Loss is: ||r||² where r should be zero for a perfect solution.
        
        Args:
            N_samples: Number of samples to evaluate
            batch_size: Batch size for evaluation
            key: Random key (uses internal key if None)
        
        Returns:
            Dict with 'mean', 'max', 'std' of squared residuals
        """
        from utils import Key
        
        if key is None:
            key = Key(self._split_key())
        elif not isinstance(key, Key):
            key = Key(key)
        
        num_batches = (N_samples + batch_size - 1) // batch_size
        all_residuals = []
        
        for batch_idx in range(num_batches):
            batch_n = min(batch_size, N_samples - batch_idx * batch_size)
            
            # Sample domain points (x only, we set t=0)
            x, _ = self.solver.sample_domain_eval(key, batch_n)
            t = np.zeros((batch_n, 1))
            
            # Get value and gradients
            if self.backend == 'jax':
                v, dv_dx, dv_dt = self._predict_with_grad_fn(jnp.array(x), jnp.array(t))
                v = np.asarray(v)
                dv_dx = np.asarray(dv_dx)
                dv_dt = np.asarray(dv_dt)
                
                # Compute Hamiltonian using JAX
                H = np.asarray(self.solver.H(jnp.array(x), jnp.array(dv_dx)))
                l = np.asarray(self.solver.l(jnp.array(x)))
            else:
                # Torch backend
                v, dv_dx, dv_dt = self._predict_with_grad_fn(x, t)
                
                # Compute Hamiltonian using solver (needs JAX arrays)
                x_jax = jnp.array(x)
                dv_dx_jax = jnp.array(dv_dx)
                H = np.asarray(self.solver.H(x_jax, dv_dx_jax))
                l = np.asarray(self.solver.l(x_jax))
            
            # VI residual: min(dV/dt + H, l - V)
            pde_term = dv_dt[:, 0, 0] + H[:, 0]  # (batch,)
            constraint_term = l[:, 0] - v[:, 0]  # (batch,)
            vi_residual = np.minimum(pde_term, constraint_term)
            
            # Squared residual
            residual_sq = vi_residual ** 2
            all_residuals.append(residual_sq)
        
        all_residuals = np.concatenate(all_residuals, axis=0)
        
        return {
            'mean': float(np.mean(all_residuals)),
            'max': float(np.max(all_residuals)),
            'std': float(np.std(np.sqrt(all_residuals)))  # std of residual (not squared)
        }
    
    def compute_volume(
        self,
        N_samples: int = 100000,
        batch_size: int = 10000,
        key=None
    ) -> Dict[str, Any]:
        """
        Estimate volume of predicted positive set as percentage.
        
        For non-PubSub cases (avoid-style): positive means safe, V(x, 0) > 0.
        For PubSub cases (reach-style): positive means target-reaching, V(x, 0) <= 0.

        Returns N_safe / N_total * 100 where N_safe stores positive-counts for
        backward compatibility of output keys.
        
        Args:
            N_samples: Number of samples for estimation
            batch_size: Batch size for evaluation
            key: Random key (uses internal key if None)
        
        Returns:
            Dict with 'volume_percentage', 'N_safe', 'N_total'
        """
        from utils import Key
        
        if key is None:
            key = Key(self._split_key())
        elif not isinstance(key, Key):
            key = Key(key)

        is_pubsub_reach = self._is_pubsub_case()
        
        num_batches = (N_samples + batch_size - 1) // batch_size
        n_safe_total = 0
        n_total = 0
        
        for batch_idx in range(num_batches):
            batch_n = min(batch_size, N_samples - batch_idx * batch_size)
            
            # Sample x from domain at t=0
            x, _ = self.solver.sample_domain_eval(key, batch_n)
            t = np.zeros((batch_n, 1))
            
            # Predict value
            v = self.predict(x, t)
            
            # Count positive points:
            # - avoid-style (default): V > 0
            # - PubSub reach-style: V <= 0
            if is_pubsub_reach:
                n_safe = int(np.sum(v[:, 0] <= 0.0))
            else:
                n_safe = int(np.sum(v[:, 0] > 0.0))
            n_safe_total += n_safe
            n_total += batch_n
        
        volume_percentage = (n_safe_total / n_total) * 100.0 if n_total > 0 else 0.0
        
        return {
            'volume_percentage': volume_percentage,
            'N_safe': n_safe_total,
            'N_total': n_total
        }
    
    def _rollout_batch_euler(self, x0_batch, key):
        """
        Roll out a batch of trajectories using Euler integration.
        
        Uses the learned value function gradient to compute optimal control/disturbance.
        Tracks whether each trajectory ever enters the unsafe set (l(x) <= 0).
        
        Args:
            x0_batch: Initial states, shape (batch, d_in)
            key: Random key
            
        Returns:
            unsafe_any: Boolean array, True if trajectory ever hit l(x) <= 0, shape (batch,)
        """
        batch_size = x0_batch.shape[0]
        dt = self.dt
        traj_len = self.traj_len
        
        # Check if initial state is already unsafe
        l_init = np.asarray(self.solver.l(jnp.array(x0_batch)))
        unsafe_any = (l_init[:, 0] <= 0.0)
        
        x = np.array(x0_batch)
        t = np.zeros((batch_size, 1))
        smooth_controls = self.solver._use_smooth_controls(training=False)
        
        for step in range(traj_len):
            # Get value function gradient for optimal control
            if self.backend == 'jax':
                _, u_x, _ = self._predict_with_grad_fn(jnp.array(x), jnp.array(t))
                u_x = jnp.array(u_x)
                x_jax = jnp.array(x)
                
                # Compute optimal control and disturbance
                u_opt = np.asarray(self.solver.u_star(x_jax, u_x, smooth=smooth_controls))
                d_opt = np.asarray(self.solver.d_star(x_jax, u_x, smooth=smooth_controls))
                
                # Dynamics: dx = f(x, u*, d*) * dt
                f_val = np.asarray(self.solver.f(x_jax, jnp.array(u_opt), jnp.array(d_opt)))
            else:
                # Torch backend
                _, u_x, _ = self._predict_with_grad_fn(x, t)
                u_x_jax = jnp.array(u_x)
                x_jax = jnp.array(x)
                
                # Compute optimal control and disturbance
                u_opt = np.asarray(self.solver.u_star(x_jax, u_x_jax, smooth=smooth_controls))
                d_opt = np.asarray(self.solver.d_star(x_jax, u_x_jax, smooth=smooth_controls))
                
                # Dynamics
                f_val = np.asarray(self.solver.f(x_jax, jnp.array(u_opt), jnp.array(d_opt)))
            
            # Euler step
            x = x + f_val * dt
            t = t + dt
            
            # Check if new state is unsafe
            l_new = np.asarray(self.solver.l(jnp.array(x)))
            unsafe_any = unsafe_any | (l_new[:, 0] <= 0.0)
        
        return unsafe_any
    
    def _rollout_batch_euler_jax(self, x0_batch, key):
        """
        JAX-optimized version of trajectory rollout using scan.
        
        Args:
            x0_batch: Initial states, shape (batch, d_in)
            key: JAX random key
            
        Returns:
            unsafe_any: Boolean array, shape (batch,)
        """
        batch_size = x0_batch.shape[0]
        dt = self.dt
        traj_len = self.traj_len
        
        # Check if initial state is already unsafe
        l_init = self.solver.l(x0_batch)
        unsafe_any_init = (l_init[:, 0] <= 0.0)
        
        # Placeholder for scan (no diffusion)
        steps = jnp.arange(traj_len)
        smooth_controls = self.solver._use_smooth_controls(training=False)
        
        def euler_step(carry, step):
            x, t, unsafe_any = carry
            
            # Get value function gradient
            _, u_x, _ = self.solver.calc_ux(self.params, x, t)
            
            # Compute optimal control and disturbance
            u_opt = self.solver.u_star(x, u_x, smooth=smooth_controls)
            d_opt = self.solver.d_star(x, u_x, smooth=smooth_controls)
            
            # Dynamics
            f_val = self.solver.f(x, u_opt, d_opt)
            
            # Euler step
            x_new = x + f_val * dt
            t_new = t + dt
            
            # Check safety
            l_new = self.solver.l(x_new)
            unsafe_any = unsafe_any | (l_new[:, 0] <= 0.0)
            
            return (x_new, t_new, unsafe_any), None
        
        t0 = jnp.zeros((batch_size, 1))
        (_, _, unsafe_any_final), _ = jax.lax.scan(
            euler_step, (x0_batch, t0, unsafe_any_init), steps
        )
        
        return unsafe_any_final
    
    def compute_f1(
        self,
        N_rollouts: int = 10000,
        batch_size: int = 1000,
        key=None
    ) -> Dict[str, Any]:
        """
        Compute F1 statistics by comparing predicted positive labels to rollout truth.
        
        Uses delta=0 threshold.
        - For non-PubSub cases (avoid-style):
          predicted positive: V(x, 0) > 0
          actual positive: trajectory never hits l(x) <= 0
        - For PubSub cases (reach-style):
          predicted positive: V(x, 0) <= 0
          actual positive: trajectory ever hits l(x) <= 0
        
        Args:
            N_rollouts: Number of trajectories to evaluate
            batch_size: Batch size for rollouts
            key: Random key (uses internal key if None)
        
        Returns:
            Dict with 'precision', 'recall', 'f1', 'accuracy', 'true_volume',
            'tp', 'fp', 'tn', 'fn'
        """
        from utils import Key
        
        # Always use delta=0 for F1 (consistent with hji_solver.evaluate_safety_metric)
        delta = 0.0
        
        if key is None:
            key = Key(self._split_key())
        elif not isinstance(key, Key):
            key = Key(key)

        is_pubsub_reach = self._is_pubsub_case()
        
        num_batches = (N_rollouts + batch_size - 1) // batch_size
        tp_total, fp_total, tn_total, fn_total = 0, 0, 0, 0
        
        # JIT compile rollout for JAX backend
        if self.backend == 'jax':
            rollout_fn = jax.jit(self._rollout_batch_euler_jax)
        
        for batch_idx in range(num_batches):
            batch_n = min(batch_size, N_rollouts - batch_idx * batch_size)
            
            # Sample initial conditions at t=0
            x0, _ = self.solver.sample_domain_t0_eval(key, batch_n)
            t0 = np.zeros((batch_n, 1))
            
            # Predicted labels at t=0
            v_pred = self.predict(np.asarray(x0), t0)
            if is_pubsub_reach:
                predicted_positive = (v_pred[:, 0] <= delta)
            else:
                predicted_positive = (v_pred[:, 0] > delta)
            
            # Roll out trajectories and get unsafe_any flag
            if self.backend == 'jax':
                unsafe_any = np.asarray(rollout_fn(x0, self._split_key()))
            else:
                unsafe_any = self._rollout_batch_euler(np.asarray(x0), key)
            
            # Ground-truth labels from rollout
            if is_pubsub_reach:
                actual_positive = unsafe_any
            else:
                actual_positive = ~unsafe_any
            
            # Confusion matrix
            tp = int(np.sum(predicted_positive & actual_positive))
            fp = int(np.sum(predicted_positive & ~actual_positive))
            tn = int(np.sum(~predicted_positive & ~actual_positive))
            fn = int(np.sum(~predicted_positive & actual_positive))
            
            tp_total += tp
            fp_total += fp
            tn_total += tn
            fn_total += fn
        
        total = tp_total + fp_total + tn_total + fn_total
        
        # Compute metrics with edge case handling
        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp_total + tn_total) / total if total > 0 else 0.0
        true_volume = (tp_total + fn_total) / total if total > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'true_volume': true_volume,
            'tp': tp_total,
            'fp': fp_total,
            'tn': tn_total,
            'fn': fn_total
        }

    def _resolve_ground_truth_problem(self) -> Optional[str]:
        """
        Resolve solver case name to optimized_dp artifact problem key.

        Returns:
            'vertical_drone', 'pursuit_evasion', 'pubsub40d', 'pubsub_nd', or None.
        """
        case = getattr(self.config, 'case', None)
        problem_map = {
            'vertical_drone': 'vertical_drone',
            'pursuit_evasion': 'pursuit_evasion',
            'pubsub_40d': 'pubsub40d',
            'pubsub_nd': 'pubsub_nd',
        }
        return problem_map.get(case)

    def _find_latest_ground_truth_artifact(
        self,
        ground_truth_dir: Path,
        problem: str,
        expected_n_subs: Optional[int] = None,
    ) -> Optional[Path]:
        """Find newest ground-truth artifact for a problem under ground_truth_dir."""
        patterns: List[str]
        if problem == "pubsub_nd":
            # Prefer exact dimension match to avoid comparing against the newest-but-wrong ns artifact.
            if expected_n_subs is not None:
                patterns = [
                    f"pubsub_nd_ns{int(expected_n_subs)}_t0_*.npz",
                    "pubsub_nd_t0_*.npz",
                ]
            else:
                patterns = [
                    "pubsub_nd_ns*_t0_*.npz",
                    "pubsub_nd_t0_*.npz",
                ]
        else:
            patterns = [f"{problem}_t0_*.npz"]

        matches: List[Path] = []
        for pattern in patterns:
            matches.extend(ground_truth_dir.glob(pattern))
        matches = sorted(set(matches), key=lambda p: p.stat().st_mtime)
        return matches[-1] if matches else None

    def compute_ground_truth_rl2_t0(
        self,
        batch_size: int = 1000,
        ground_truth_dir: Union[str, Path] = "examples/ground_truth",
        artifact_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Compute ground-truth error metrics at t=0 against optimized_dp artifact.

        Returned metrics:
            - mse_t0: mean squared error
            - rmse_t0: root mean squared error
            - rl2_t0: relative L2 error

        RL2 is defined as:
            sqrt(sum((u_ref - u_pred)^2) / sum(u_ref^2))

        Supported cases:
            - vertical_drone
            - pursuit_evasion
            - pubsub_40d / pubsub_nd (all-equal ND slice from PubSub artifacts)

        If unsupported or artifact is missing/invalid, warns and returns None metrics.
        """
        problem = self._resolve_ground_truth_problem()
        if problem is None:
            return {'mse_t0': None, 'rmse_t0': None, 'rl2_t0': None, 'num_points': None, 'artifact': None}

        batch_size = max(1, int(batch_size))

        if artifact_path is not None:
            artifact = Path(artifact_path)
            if not artifact.exists():
                warnings.warn(
                    f"Ground truth artifact not found: {artifact}. Skipping t=0 ground-truth metrics.",
                    UserWarning,
                    stacklevel=2,
                )
                return {'mse_t0': None, 'rmse_t0': None, 'rl2_t0': None, 'num_points': None, 'artifact': None}
        else:
            gt_dir = Path(ground_truth_dir)
            expected_n_subs = int(self.d_in) - 1 if problem == "pubsub_nd" else None
            artifact = self._find_latest_ground_truth_artifact(
                gt_dir,
                problem,
                expected_n_subs=expected_n_subs,
            )
            if artifact is None:
                if problem == "pubsub_nd" and expected_n_subs is not None:
                    warnings.warn(
                        f"No ground truth artifact found for '{problem}' with n_subs={expected_n_subs} "
                        f"under {gt_dir}. Skipping t=0 ground-truth metrics.",
                        UserWarning,
                        stacklevel=2,
                    )
                    return {'mse_t0': None, 'rmse_t0': None, 'rl2_t0': None, 'num_points': None, 'artifact': None}
                warnings.warn(
                    f"No ground truth artifact found for '{problem}' under {gt_dir}. Skipping t=0 ground-truth metrics.",
                    UserWarning,
                    stacklevel=2,
                )
                return {'mse_t0': None, 'rmse_t0': None, 'rl2_t0': None, 'num_points': None, 'artifact': None}

        try:
            with np.load(artifact) as data:
                if "value_t0" not in data.files:
                    raise KeyError("value_t0")
                value_t0 = np.asarray(data["value_t0"], dtype=np.float64)

                if problem == "vertical_drone":
                    for key in ("z_axis", "vz_axis"):
                        if key not in data.files:
                            raise KeyError(key)
                    z_axis = np.asarray(data["z_axis"], dtype=np.float64)
                    vz_axis = np.asarray(data["vz_axis"], dtype=np.float64)
                    n_z = z_axis.shape[0]
                    n_vz = vz_axis.shape[0]
                    if value_t0.shape != (n_z, n_vz):
                        raise ValueError(
                            f"value_t0 shape {value_t0.shape} does not match axes {(n_z, n_vz)}"
                        )

                    total_points = n_z * n_vz
                    value_flat = value_t0.reshape(-1)
                    ref_sq_sum = float(np.sum(value_flat ** 2))
                    sq_err_sum = 0.0
                    for start in range(0, total_points, batch_size):
                        end = min(start + batch_size, total_points)
                        idx = np.arange(start, end, dtype=np.int64)
                        i_z = idx // n_vz
                        i_vz = idx % n_vz
                        x_batch = np.column_stack([z_axis[i_z], vz_axis[i_vz]])
                        t_batch = np.zeros((end - start, 1), dtype=np.float64)
                        pred = self.predict(x_batch, t_batch)[:, 0]
                        err = pred - value_flat[start:end]
                        sq_err_sum += float(np.sum(err ** 2))

                elif problem == "pursuit_evasion":
                    for key in ("x1_axis", "x2_axis", "theta_axis"):
                        if key not in data.files:
                            raise KeyError(key)
                    x1_axis = np.asarray(data["x1_axis"], dtype=np.float64)
                    x2_axis = np.asarray(data["x2_axis"], dtype=np.float64)
                    theta_axis = np.asarray(data["theta_axis"], dtype=np.float64)
                    n_x1 = x1_axis.shape[0]
                    n_x2 = x2_axis.shape[0]
                    n_theta = theta_axis.shape[0]
                    if value_t0.shape != (n_x1, n_x2, n_theta):
                        raise ValueError(
                            f"value_t0 shape {value_t0.shape} does not match axes {(n_x1, n_x2, n_theta)}"
                        )

                    total_points = n_x1 * n_x2 * n_theta
                    stride_x1 = n_x2 * n_theta
                    stride_x2 = n_theta
                    value_flat = value_t0.reshape(-1)
                    ref_sq_sum = float(np.sum(value_flat ** 2))
                    sq_err_sum = 0.0
                    for start in range(0, total_points, batch_size):
                        end = min(start + batch_size, total_points)
                        idx = np.arange(start, end, dtype=np.int64)
                        i_x1 = idx // stride_x1
                        rem = idx % stride_x1
                        i_x2 = rem // stride_x2
                        i_theta = rem % stride_x2
                        x_batch = np.column_stack(
                            [x1_axis[i_x1], x2_axis[i_x2], theta_axis[i_theta]]
                        )
                        t_batch = np.zeros((end - start, 1), dtype=np.float64)
                        pred = self.predict(x_batch, t_batch)[:, 0]
                        err = pred - value_flat[start:end]
                        sq_err_sum += float(np.sum(err ** 2))

                elif problem in ("pubsub40d", "pubsub_nd"):
                    for key in ("x0_axis", "xi_axis"):
                        if key not in data.files:
                            raise KeyError(key)
                    x0_axis = np.asarray(data["x0_axis"], dtype=np.float64)
                    xi_axis = np.asarray(data["xi_axis"], dtype=np.float64)
                    n_x0 = x0_axis.shape[0]
                    n_xi = xi_axis.shape[0]
                    if value_t0.shape != (n_x0, n_xi):
                        raise ValueError(
                            f"value_t0 shape {value_t0.shape} does not match axes {(n_x0, n_xi)}"
                        )

                    n_subs_model = int(self.d_in) - 1
                    if "n_subs" in data.files:
                        n_subs_artifact = int(np.asarray(data["n_subs"]).item())
                        if n_subs_artifact != n_subs_model:
                            raise ValueError(
                                "PubSub ground truth n_subs mismatch: "
                                f"artifact n_subs={n_subs_artifact}, model n_subs={n_subs_model}"
                            )
                    elif problem == "pubsub_nd":
                        raise KeyError("n_subs")

                    total_points = n_x0 * n_xi
                    value_flat = value_t0.reshape(-1)
                    ref_sq_sum = float(np.sum(value_flat ** 2))
                    sq_err_sum = 0.0
                    for start in range(0, total_points, batch_size):
                        end = min(start + batch_size, total_points)
                        idx = np.arange(start, end, dtype=np.int64)
                        i_x0 = idx // n_xi
                        i_xi = idx % n_xi

                        x0_vals = x0_axis[i_x0]
                        xi_vals = xi_axis[i_xi]
                        x_batch = np.zeros((end - start, self.d_in), dtype=np.float64)
                        x_batch[:, 0] = x0_vals
                        x_batch[:, 1:] = xi_vals[:, np.newaxis]
                        t_batch = np.zeros((end - start, 1), dtype=np.float64)
                        pred = self.predict(x_batch, t_batch)[:, 0]
                        err = pred - value_flat[start:end]
                        sq_err_sum += float(np.sum(err ** 2))
                else:
                    return {'mse_t0': None, 'rmse_t0': None, 'rl2_t0': None, 'num_points': None, 'artifact': None}

            if total_points > 0:
                mse = float(sq_err_sum / float(total_points))
                rmse = float(np.sqrt(mse))
                if ref_sq_sum <= 1e-18:
                    rl2 = 0.0 if sq_err_sum <= 1e-18 else float("inf")
                else:
                    rl2 = float(np.sqrt(sq_err_sum / ref_sq_sum))
            else:
                mse = None
                rmse = None
                rl2 = None
            return {
                'mse_t0': mse,
                'rmse_t0': rmse,
                'rl2_t0': rl2,
                'num_points': int(total_points),
                'artifact': str(artifact),
            }
        except (KeyError, ValueError, OSError) as exc:
            warnings.warn(
                f"Failed ground truth metrics evaluation with artifact {artifact}: {exc}. Skipping t=0 ground-truth metrics.",
                UserWarning,
                stacklevel=2,
            )
            return {'mse_t0': None, 'rmse_t0': None, 'rl2_t0': None, 'num_points': None, 'artifact': str(artifact)}
    
    def evaluate(
        self,
        N_rollouts: int = 10000,
        N_residual_samples: int = 10000,
        N_volume_samples: int = 100000,
        batch_size: int = 1000,
        verbose: bool = True,
        ground_truth_dir: Union[str, Path] = "examples/ground_truth",
        ground_truth_artifact: Optional[Union[str, Path]] = None,
    ) -> ModelEvalResult:
        """
        Perform comprehensive model evaluation.
        
        All metrics use threshold delta=0 (zero-level set) for consistency
        with hji_solver.evaluate_safety_metric.
        
        Args:
            N_rollouts: Number of trajectories for F1 computation
            N_residual_samples: Number of samples for PDE residual
            N_volume_samples: Number of samples for volume estimation
            batch_size: Batch size for all computations
            verbose: Print progress information
            ground_truth_dir: Directory to search for ground-truth artifacts
            ground_truth_artifact: Optional explicit path to a specific artifact
        
        Returns:
            ModelEvalResult containing all evaluation metrics
        """
        if verbose:
            print(f"Evaluating model (backend={self.backend})...")
            print(f"  N_rollouts={N_rollouts}, "
                  f"N_residual={N_residual_samples}, N_volume={N_volume_samples}")
        
        # Compute F1 statistics (always uses delta=0)
        if verbose:
            print("  Computing F1 statistics (delta=0)...")
        f1_result = self.compute_f1(N_rollouts, batch_size)
        
        # Compute volume (always uses delta=0)
        if verbose:
            print("  Computing volume estimation (delta=0)...")
        volume_result = self.compute_volume(N_volume_samples, batch_size)
        
        # Compute PDE residual
        if verbose:
            print("  Computing PDE residual...")
        residual_result = self.compute_pde_residual(N_residual_samples, batch_size)

        # Compute t=0 ground-truth metrics (if supported artifact exists)
        if verbose:
            print("  Computing ground-truth metrics at t=0 (if available)...")
        ground_truth_result = self.compute_ground_truth_rl2_t0(
            batch_size=batch_size,
            ground_truth_dir=ground_truth_dir,
            artifact_path=ground_truth_artifact,
        )
        
        result = ModelEvalResult(
            # F1 statistics
            precision=f1_result['precision'],
            recall=f1_result['recall'],
            f1=f1_result['f1'],
            accuracy=f1_result['accuracy'],
            true_volume=f1_result['true_volume'],
            tp=f1_result['tp'],
            fp=f1_result['fp'],
            tn=f1_result['tn'],
            fn=f1_result['fn'],
            # Volume estimation
            volume_percentage=volume_result['volume_percentage'],
            N_safe=volume_result['N_safe'],
            N_total_volume=volume_result['N_total'],
            # PDE residual
            pde_residual_mean=residual_result['mean'],
            pde_residual_max=residual_result['max'],
            pde_residual_std=residual_result['std'],
            # Ground truth comparison
            ground_truth_mse_t0=ground_truth_result['mse_t0'],
            ground_truth_rmse_t0=ground_truth_result['rmse_t0'],
            ground_truth_rl2_t0=ground_truth_result['rl2_t0'],
            ground_truth_num_points=ground_truth_result['num_points'],
            ground_truth_artifact=ground_truth_result['artifact'],
            # Evaluation parameters
            N_rollouts=N_rollouts,
            N_residual_samples=N_residual_samples,
            backend=self.backend
        )
        
        if verbose:
            print("  Evaluation complete!")
            print(result)
        
        return result
