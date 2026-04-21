"""
Example script for training/evaluating the 13D quadcopter HJI model.
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
from problems import Quadcopter13D_HJI

LOSS_CHOICES = ("vipinns", "pinns", "fspinns", "fspinnsbatched")

FIXED_IC = {
    "p_z": 0.54,
    "q_w": 0.44,
    "q_x": -0.45,
    "q_y": 0.27,
    "q_z": -0.73,
    "v_x": 5.00,
    "v_y": -1.07,
    "v_z": -3.34,
    "omega_x": 3.19,
    "omega_y": -2.80,
    "omega_z": 3.43,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="13D Quadcopter HJI training script")
    parser.add_argument("-l", "--loss", type=str, choices=LOSS_CHOICES, default="vipinns")
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
    return parser


def configure_training(config, args):
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
    config2.loss_method = args.loss
    config2.smooth_control = config.smooth_control
    config2.smooth_control_scope = config.smooth_control_scope
    config2.smooth_control_map = config.smooth_control_map
    config2.smooth_control_rho = rho_schedule[1]

    config3 = config.get_train_config()
    config3.lr = lrs[2]
    config3.iter = iters[2]
    config3.loss_method = args.loss
    config3.smooth_control = config.smooth_control
    config3.smooth_control_scope = config.smooth_control_scope
    config3.smooth_control_map = config.smooth_control_map
    config3.smooth_control_rho = rho_schedule[2]

    return config2, config3


def visualize_results(solver, params, loss_method, run_path=None):
    import matplotlib.pyplot as plt

    x_axis = jnp.linspace(-3.0, 3.0, 120)
    y_axis = jnp.linspace(-3.0, 3.0, 120)
    x_grid, y_grid = jnp.meshgrid(x_axis, y_axis)

    n = x_grid.size
    x_eval = jnp.stack(
        [
            x_grid.reshape(-1),
            y_grid.reshape(-1),
            jnp.ones(n) * FIXED_IC["p_z"],
            jnp.ones(n) * FIXED_IC["q_w"],
            jnp.ones(n) * FIXED_IC["q_x"],
            jnp.ones(n) * FIXED_IC["q_y"],
            jnp.ones(n) * FIXED_IC["q_z"],
            jnp.ones(n) * FIXED_IC["v_x"],
            jnp.ones(n) * FIXED_IC["v_y"],
            jnp.ones(n) * FIXED_IC["v_z"],
            jnp.ones(n) * FIXED_IC["omega_x"],
            jnp.ones(n) * FIXED_IC["omega_y"],
            jnp.ones(n) * FIXED_IC["omega_z"],
        ],
        axis=-1,
    )

    t_eval = jnp.zeros((n, 1))
    v_pred = np.asarray(solver.calc_u(params, x_eval, t_eval)).reshape(120, 120)
    l_val = np.asarray(solver.l(x_eval)).reshape(120, 120)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    ax = axes[0]
    im = ax.imshow(v_pred, extent=(-3, 3, -3, 3), origin="lower", cmap="RdBu", aspect="equal")
    ax.contour(x_axis, y_axis, v_pred, levels=[0.0], colors="black", linewidths=2)
    ax.add_patch(plt.Circle((0, 0), solver.r_0, fill=False, linestyle="--", color="green", linewidth=1.8))
    ax.set_title(f"V(px, py, t=0) [{loss_method}]")
    ax.set_xlabel("px")
    ax.set_ylabel("py")
    plt.colorbar(im, ax=ax, label="V")

    ax = axes[1]
    im = ax.imshow(l_val, extent=(-3, 3, -3, 3), origin="lower", cmap="RdBu", aspect="equal")
    ax.contour(x_axis, y_axis, l_val, levels=[0.0], colors="black", linewidths=2)
    ax.add_patch(plt.Circle((0, 0), solver.r_0, fill=False, linestyle="--", color="green", linewidth=1.8))
    ax.set_title("Terminal Level Set l(x)")
    ax.set_xlabel("px")
    ax.set_ylabel("py")
    plt.colorbar(im, ax=ax, label="l")

    ax = axes[2]
    im = ax.imshow((v_pred >= 0).astype(float), extent=(-3, 3, -3, 3), origin="lower", cmap="RdYlGn", aspect="equal")
    ax.contour(x_axis, y_axis, v_pred, levels=[0.0], colors="black", linewidths=2)
    ax.add_patch(plt.Circle((0, 0), solver.r_0, fill=False, linestyle="--", color="blue", linewidth=1.8))
    ax.set_title("Predicted Safe Set (V >= 0)")
    ax.set_xlabel("px")
    ax.set_ylabel("py")
    plt.colorbar(im, ax=ax, label="safe")

    plt.tight_layout()
    save_path = os.path.join(run_path, f"quadcopter_result_{loss_method}.png") if run_path else f"quadcopter_result_{loss_method}.png"
    plt.savefig(save_path, dpi=150)
    if solver.config.save_to_wandb:
        wandb.summary["quadcopter_plot"] = wandb.Image(fig)
    plt.close(fig)


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.float == 1:
        jax.config.update("jax_enable_x64", True)

    config = Quadcopter13D_HJI.get_base_config(traj_len=args.disc)
    if args.sigma_noise is not None:
        config.sigma_noise = args.sigma_noise

    config.d_hidden = args.d_hidden
    config.num_layers = 5
    config.activation = args.activation
    config.batch_pde = args.batch_pde
    config.batch_traj = args.batch_traj
    config.periodic = False
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
    config.hard_constraint_type = args.hard_constraint_type
    config.safety_eval = bool(args.safety)
    config.num_safety_rollouts = args.num_rollouts
    config.safety_rollout_batch_size = 500
    config.loss_method = args.loss
    config.auto_close = False

    config2, config3 = configure_training(config, args)

    solver = Quadcopter13D_HJI(config)
    if args.tag is not None and config.save_to_wandb:
        solver.wandb_tags(args.tag)

    controller = HJI_Controller(solver, seed=args.seed)
    controller.append_train_config(config2)
    controller.append_train_config(config3)
    controller.solve()

    run_name = args.run_name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"quadcopter_{args.loss}_{timestamp}"

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
