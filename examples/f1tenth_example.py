"""
Example script for training/evaluating the 7D F1Tenth HJI model.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import argparse
import json
import os
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import wandb

from evaluation import ModelEvaluator
from hji_solver import HJI_Controller
from problems import F1tenth_HJI

LOSS_CHOICES = ("vipinns", "pinns", "fspinns", "fspinnsbatched")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="F1Tenth HJI training script")
    parser.add_argument("-l", "--loss", type=str, choices=LOSS_CHOICES, default="vipinns")
    parser.add_argument("--loss2", type=str, choices=LOSS_CHOICES, default=None)
    parser.add_argument("--loss3", type=str, choices=LOSS_CHOICES, default=None)
    parser.add_argument("-f", "--float", type=int, default=1, help="0=f32, 1=f64")
    parser.add_argument("-d", "--disc", type=int, default=50)
    parser.add_argument("-w", "--wandb", type=int, default=0)
    parser.add_argument("-i", "--iter", type=int, default=20000)
    parser.add_argument("-t", "--test", action="store_true")
    parser.add_argument("-c", "--constraint", type=int, default=1)
    parser.add_argument("-tc", "--terminal_constraint", type=int, default=0)
    parser.add_argument("--alternative_tc", type=int, default=0)
    parser.add_argument("-s", "--safety", type=int, default=1)
    parser.add_argument("-n", "--num_rollouts", type=int, default=1_000_000)
    parser.add_argument("-tg", "--term_grad_loss", type=int, default=0)
    parser.add_argument("-sg", "--stop_grad", type=int, default=0)
    parser.add_argument("-sc", "--smooth_control", type=int, default=0)
    parser.add_argument("-tag", "--tag", type=str, default=None)
    parser.add_argument("-ht", "--hard_constraint_type", type=str, default="quadratic")
    parser.add_argument("-rn", "--run_name", type=str, default=None)
    parser.add_argument("-e", "--evaluate", type=int, default=1)
    parser.add_argument("-en", "--eval_rollouts", type=int, default=1_000_000)
    parser.add_argument("-sn", "--sigma_noise", type=float, default=None)
    parser.add_argument("--rad", type=int, default=0)
    parser.add_argument("--rad_candidate_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--activation", type=str, choices=["waveact", "tanh", "swish", "sine"], default="swish")
    parser.add_argument("--d_hidden", type=int, default=128)
    parser.add_argument("--batch_pde", type=int, default=4096)
    parser.add_argument("--batch_traj", type=int, default=128)
    parser.add_argument("--input_normalization", type=int, default=0)
    parser.add_argument("--bound_rollout_states", type=int, default=None)
    parser.add_argument("--f1_map_path", type=str, default=None)
    return parser


def configure_training(config, args, stage2_loss, stage3_loss):
    if args.test:
        iters = [100, 50, 50]
        lrs = [1e-3, 1e-4, 1e-5]
    else:
        iters = [args.iter, args.iter // 2, args.iter // 2]
        lrs = [1e-3, 1e-4, 1e-5]

    rho_schedule = [1e-2, 1e-3, 1e-4]

    config.additional_losses = True
    config.lr = lrs[0]
    config.iter = iters[0]
    config.smooth_control_rho = rho_schedule[0]

    config2 = config.get_train_config()
    config2.lr = lrs[1]
    config2.iter = iters[1]
    config2.loss_method = stage2_loss
    config2.smooth_control = config.smooth_control
    config2.smooth_control_scope = config.smooth_control_scope
    config2.smooth_control_map = config.smooth_control_map
    config2.smooth_control_rho = rho_schedule[1]

    config3 = config.get_train_config()
    config3.lr = lrs[2]
    config3.iter = iters[2]
    config3.loss_method = stage3_loss
    config3.smooth_control = config.smooth_control
    config3.smooth_control_scope = config.smooth_control_scope
    config3.smooth_control_map = config.smooth_control_map
    config3.smooth_control_rho = rho_schedule[2]

    return config2, config3


def visualize_results(solver, params, loss_method, run_path=None):
    import matplotlib.pyplot as plt

    nx, ny = 140, 112
    x_axis = np.linspace(float(solver.x_min), float(solver.x_max), nx)
    y_axis = np.linspace(float(solver.y_min), float(solver.y_max), ny)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)

    n = x_grid.size
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    x_eval = jnp.zeros((n, 7), dtype=dtype)
    x_eval = x_eval.at[:, 0].set(jnp.asarray(x_grid.reshape(-1)))
    x_eval = x_eval.at[:, 1].set(jnp.asarray(y_grid.reshape(-1)))
    x_eval = x_eval.at[:, 2].set(0.0)
    x_eval = x_eval.at[:, 3].set(max(float(solver.v_min), 2.0))
    x_eval = x_eval.at[:, 4].set(0.0)
    x_eval = x_eval.at[:, 5].set(0.0)
    x_eval = x_eval.at[:, 6].set(0.0)
    t_eval = jnp.zeros((n, 1), dtype=dtype)

    v_pred = np.asarray(solver.calc_u(params, x_eval, t_eval)).reshape(ny, nx)
    l_map = np.asarray(solver.l(x_eval)).reshape(ny, nx)

    _, ux, _ = solver.calc_ux(params, x_eval, t_eval)
    u_opt = np.asarray(solver.u_star(x_eval, ux))
    steer_rate = u_opt[:, 0].reshape(ny, nx)
    accel = u_opt[:, 1].reshape(ny, nx)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    extent = (solver.x_min, solver.x_max, solver.y_min, solver.y_max)

    ax = axes[0, 0]
    im = ax.imshow(v_pred, extent=extent, origin="lower", cmap="RdBu")
    ax.contour(x_axis, y_axis, v_pred, levels=[0.0], colors="black", linewidths=1.8)
    ax.set_title(f"V(x, y, t=0) [{loss_method}]")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="V")

    ax = axes[0, 1]
    im = ax.imshow(l_map, extent=extent, origin="lower", cmap="RdBu")
    ax.contour(x_axis, y_axis, l_map, levels=[0.0], colors="black", linewidths=1.8)
    ax.set_title("Terminal Level Set l(x)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="l")

    ax = axes[1, 0]
    im = ax.imshow(steer_rate, extent=extent, origin="lower", cmap="coolwarm", vmin=solver.sv_min, vmax=solver.sv_max)
    ax.contour(x_axis, y_axis, v_pred, levels=[0.0], colors="black", linewidths=1.2)
    ax.set_title("Steering Rate Control")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="u[0]")

    ax = axes[1, 1]
    im = ax.imshow(accel, extent=extent, origin="lower", cmap="coolwarm", vmin=-solver.a_max, vmax=solver.a_max)
    ax.contour(x_axis, y_axis, v_pred, levels=[0.0], colors="black", linewidths=1.2)
    ax.set_title("Acceleration Control")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="u[1]")

    plt.tight_layout()
    save_path = os.path.join(run_path, f"f1tenth_result_{loss_method}.png") if run_path else f"f1tenth_result_{loss_method}.png"
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    if solver.config.save_to_wandb:
        wandb.summary["f1tenth_plot"] = wandb.Image(fig)
    plt.close(fig)


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.float == 1:
        jax.config.update("jax_enable_x64", True)

    stage1_loss = args.loss
    stage2_loss = args.loss2 or stage1_loss
    stage3_loss = args.loss3 or stage1_loss

    config = F1tenth_HJI.get_base_config(traj_len=args.disc)
    if args.sigma_noise is not None:
        config.sigma_noise = args.sigma_noise
    if args.f1_map_path is not None:
        config.f1_map_path = args.f1_map_path

    config.d_hidden = args.d_hidden
    config.num_layers = 5
    config.activation = args.activation
    config.batch_pde = args.batch_pde
    config.batch_ic = 2048
    config.batch_traj = args.batch_traj
    config.periodic = True
    config.periodic_idx = (4,)
    config.save_to_wandb = bool(args.wandb)
    config.random_sample = True
    config.hard_constraint = bool(args.constraint)
    config.terminal_hard_constraint = bool(args.terminal_constraint)
    config.alternative_tc = bool(args.alternative_tc)
    config.term_grad_loss = bool(args.term_grad_loss)
    config.stop_grad = bool(args.stop_grad)
    config.smooth_control = bool(args.smooth_control)
    config.input_normalization = bool(args.input_normalization)
    config.smooth_control_scope = "training_only"
    config.smooth_control_map = "sqrt"
    config.use_rad_sampling = bool(args.rad)
    if args.rad_candidate_size is not None:
        config.rad_candidate_size = int(args.rad_candidate_size)
    if args.bound_rollout_states is not None:
        config.bound_rollout_states = bool(args.bound_rollout_states)
    config.hard_constraint_type = args.hard_constraint_type
    config.safety_eval = bool(args.safety)
    config.num_safety_rollouts = args.num_rollouts
    config.safety_rollout_batch_size = 500
    config.loss_method = stage1_loss
    config.auto_close = False

    config2, config3 = configure_training(config, args, stage2_loss, stage3_loss)

    solver = F1tenth_HJI(config)
    if args.tag is not None and config.save_to_wandb:
        solver.wandb_tags(args.tag)

    controller = HJI_Controller(solver, seed=args.seed)
    controller.append_train_config(config2)
    controller.append_train_config(config3)
    controller.solve()

    run_name = args.run_name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"f1tenth_{args.loss}_{timestamp}"

    run_path = solver.save_model(params=controller.params, run_name=run_name, save_dir="./runs")

    if args.evaluate:
        evaluator = ModelEvaluator(solver, params=controller.params)
        result = evaluator.evaluate(
            N_rollouts=args.eval_rollouts,
            N_residual_samples=args.eval_rollouts,
            N_volume_samples=args.eval_rollouts * 10,
            batch_size=500,
            verbose=True,
        )
        eval_path = os.path.join(run_path, "eval_result.json")
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        if config.save_to_wandb:
            wandb.summary.update(result.to_dict())

    visualize_results(solver, controller.params, args.loss, run_path)
    controller.close()

    if args.float == 1:
        jax.config.update("jax_enable_x64", False)


if __name__ == "__main__":
    main()
