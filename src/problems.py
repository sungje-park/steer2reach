from config import Config
from utils import *
import jax
from jax import numpy as jnp
import numpy as np
import wandb
from pathlib import Path
from hji_solver import HJI_Solver


class VerticalDrone_HJI(HJI_Solver):
    """
    2D Vertical Drone HJI Reachability Problem.
    
    State: x = (z, v_z) where z is height, v_z is vertical velocity.
    The drone must stay within a safe height band z ∈ [0, 3].
    
    Dynamics:
        z_dot = v_z
        v_z_dot = K*u - g
    
    where:
        - z ∈ [-0.5, 3.5] is the height
        - v_z ∈ [-4, 4] is the vertical velocity
        - g = 9.8 is gravitational acceleration
        - K = 12 is the control gain
        - u ∈ [-1, 1] is the control input (thrust effort)
    
    Failure set: |z - 1.5| >= 1.5 (i.e., z <= 0 or z >= 3)
    Signed distance: l(x) = 1.5 - |z - 1.5| (positive inside safe set)
    
    Hamiltonian (maximized over u):
        H(x, p) = p1 * v_z + p2 * (K*u* - g)
                = p1 * v_z - p2 * g + |p2 * K|
    
    Optimal control: u* = sign(p2 * K) = sign(p2) (since K > 0)
    """
    
    def __init__(self, config: Config):
        super().__init__(config)
        # Drone parameters
        self.g = getattr(config, 'g', 9.8)       # Gravitational acceleration
        self.K = getattr(config, 'K', 12.0)      # Control gain
        self.z_center = getattr(config, 'z_center', 1.5)  # Center of safe region
        self.z_half_width = getattr(config, 'z_half_width', 1.5)  # Half-width of safe region
    
    def f(self, x, u, d):
        """
        Drone dynamics: dx/dt = f(x, u)
        Note: d (disturbance) is not used in this problem.
        
        Args:
            x: State [z, v_z], shape (batch, 2)
            u: Control input, shape (batch, 1)
            d: Disturbance (unused), shape (batch, 1)
        
        Returns:
            dx/dt: shape (batch, 2)
        """
        def dynamics(x_i, u_i, d_i):
            z_dot = x_i[1]  # v_z
            v_z_dot = self.K * u_i[0] - self.g
            return jnp.array([z_dot, v_z_dot])
        
        return jax.vmap(dynamics, in_axes=(0, 0, 0))(x, u, d)
    
    def l(self, x):
        """
        Signed distance function (batched).
        l(x) = z_half_width - |z - z_center|
        
        Positive inside safe set, negative outside.
        Safe set: z ∈ [z_center - z_half_width, z_center + z_half_width] = [0, 3]
        
        Args:
            x: State [z, v_z], shape (batch, 2)
        
        Returns:
            Signed distance, shape (batch, 1)
        """
        z = x[..., 0:1]
        return self.z_half_width - jnp.abs(z - self.z_center)
    
    def u_star(self, x, dv, *, smooth: bool = False):
        """
        Optimal control (maximizer of Hamiltonian).
        u* = sign(p2 * K) = sign(p2) since K > 0
        
        The term p2 * K * u is maximized when u = sign(p2 * K).
        
        Args:
            x: State, shape (batch, 2)
            dv: Gradient of value function, shape (batch, d_out, d_in) = (batch, 1, 2)
        
        Returns:
            Optimal control, shape (batch, 1)
        """
        dv_flat = dv[:, 0, :]  # (batch, 2)
        p2 = dv_flat[:, 1]  # Gradient w.r.t. v_z
        if smooth:
            u_min, u_max, _ = self.get_control_bounds()
            s = self.smooth_sign(p2)
            return self.map_to_bounds(s, u_min, u_max)[:, jnp.newaxis]
        return jnp.sign(p2)[:, jnp.newaxis]
    
    def d_star(self, x, dv, *, smooth: bool = False):
        """
        Optimal disturbance (minimizer).
        For this problem, there is no disturbance, so return zeros.
        
        Args:
            x: State, shape (batch, 2)
            dv: Gradient of value function, shape (batch, 1, 2)
        
        Returns:
            Zero disturbance, shape (batch, 1)
        """
        return jnp.zeros((x.shape[0], 1))
    
    def get_control_bounds(self):
        """
        Control bounds for vertical drone: u ∈ [-1, 1]
        """
        return -1.0, 1.0, 1
    
    def sample_domain(self, key: Key, batch_size):
        """Sample from state space for training."""
        # z ∈ [-0.5, 3.5]
        z = jax.random.uniform(key.newkey(), (batch_size, 1), minval=-0.5, maxval=3.5)
        # v_z ∈ [-4, 4]
        v_z = jax.random.uniform(key.newkey(), (batch_size, 1), minval=-4.0, maxval=4.0)
        x_pde = jnp.concatenate([z, v_z], axis=-1)
        t_pde = jax.random.uniform(key.newkey(), (batch_size, 1),
                                   minval=self.config.t_range[0],
                                   maxval=self.config.t_range[1])
        return x_pde, t_pde
    
    @staticmethod
    def get_base_config(traj_len=50):
        """Get configuration for vertical drone problem."""
        T = 1.2
        config = Config(
            case='vertical_drone',
            d_in=2,
            d_out=1,
            traj_len=traj_len,
            delta_t=T / traj_len,
            x_range=((-0.5, 3.5), (-4.0, 4.0)),
            t_range=(0, T),
            periodic=False,
            random_sample=True,
        )
        # Add drone-specific attributes
        config.g = 9.8
        config.K = 12.0
        config.z_center = 1.5
        config.z_half_width = 1.5
        config.sigma_noise = 0.01
        return config


class PursuitEvasion_HJI(HJI_Solver):
    """
    Pursuit-Evasion Game (Air3D) - a classic HJI reachability problem.
    
    State: x = (x1, x2, x3) where (x1, x2) is relative position, x3 is relative heading.
    The evader tries to stay outside a capture radius, while the pursuer tries to capture.
    
    Dynamics:
        x1_dot = -v_e + v_p * cos(x3) + u * x2
        x2_dot = v_p * sin(x3) - u * x1  
        x3_dot = d - u
    
    where u is evader angular velocity, d is pursuer angular velocity.
    """
    
    def __init__(self, config: Config):
        super().__init__(config)
        # Game parameters
        self.v_e = getattr(config, 'v_e', 0.75)      # Evader speed
        self.v_p = getattr(config, 'v_p', 0.75)      # Pursuer speed
        self.beta = getattr(config, 'beta', 0.25)   # Capture radius
        self.omega = getattr(config, 'omega', 3.0)  # Max turn rate
    
    def f(self, x, u, d):
        """Pursuit-evasion dynamics in relative coordinates (matching original notebook)."""
        def dynamics(x_i, u_i, d_i):
            x1_dot = -self.v_e + self.v_p * jnp.cos(x_i[2]) + u_i[0] * x_i[1]
            x2_dot = self.v_p * jnp.sin(x_i[2]) - u_i[0] * x_i[0]
            x3_dot = d_i[0] - u_i[0]
            return jnp.array([x1_dot, x2_dot, x3_dot])
        
        return jax.vmap(dynamics, in_axes=(0, 0, 0))(x, u, d)
    
    def l(self, x):
        """Signed distance to capture set: l(x) = ||(x1, x2)|| - beta (batched)"""
        return jnp.sqrt(x[..., 0:1] ** 2 + x[..., 1:2] ** 2) - self.beta
    
    # def _l_single(self, x):
    #     """Signed distance for a single (unbatched) input."""
    #     return jnp.sqrt(x[0] ** 2 + x[1] ** 2) - self.beta
    
    def u_star(self, x, dv, *, smooth: bool = False):
        """
        Optimal evader control (maximizer).
        u* = omega * sign(p1*x2 - p2*x1 - p3)
        """
        dv_flat = dv[:, 0, :]  # (batch, d_in)
        p1, p2, p3 = dv_flat[:, 0], dv_flat[:, 1], dv_flat[:, 2]
        x1, x2 = x[:, 0], x[:, 1]
        a = p1 * x2 - p2 * x1 - p3
        if smooth:
            s = self.smooth_sign(a)
            return self.map_to_bounds(s, -self.omega, self.omega)[:, jnp.newaxis]
        return (self.omega * jnp.sign(a))[:, jnp.newaxis]
    
    def d_star(self, x, dv, *, smooth: bool = False):
        """
        Optimal pursuer control (minimizer).
        d* = -omega * sign(p3)
        """
        dv_flat = dv[:, 0, :]  # (batch, d_in)
        p3 = dv_flat[:, 2]
        if smooth:
            s = -self.smooth_sign(p3)
            return self.map_to_bounds(s, -self.omega, self.omega)[:, jnp.newaxis]
        return (-self.omega * jnp.sign(p3))[:, jnp.newaxis]
    
    def get_control_bounds(self):
        """
        Control bounds for pursuit-evasion: u ∈ [-omega, omega]
        """
        return -self.omega, self.omega, 1
    
    def sample_domain(self, key: Key, batch_size):
        """Sample from state space with periodic angle."""
        x1 = jax.random.uniform(key.newkey(), (batch_size, 1), minval=-1, maxval=1)
        x2 = jax.random.uniform(key.newkey(), (batch_size, 1), minval=-1, maxval=1)
        x3 = jax.random.uniform(key.newkey(), (batch_size, 1), minval=-jnp.pi, maxval=jnp.pi)
        x_pde = jnp.concatenate([x1, x2, x3], axis=-1)
        t_pde = jax.random.uniform(key.newkey(), (batch_size, 1),
                                   minval=self.config.t_range[0],
                                   maxval=self.config.t_range[1])
        return x_pde, t_pde
    
    @staticmethod
    def get_base_config(traj_len=50):
        """Get configuration for pursuit-evasion problem."""
        T = 1
        config = Config(
            case='pursuit_evasion',
            d_in=3,
            d_out=1,
            traj_len=traj_len,
            delta_t=T / traj_len,
            x_range=((-1, 1), (-1, 1), (-jnp.pi, jnp.pi)),
            t_range=(0, T),
            periodic=True
        )
        # Add HJI-specific attributes
        config.v_e = 0.75
        config.v_p = 0.75
        config.beta = 0.25
        config.omega = 3.0
        config.sigma_noise = 0.01
        return config


class Quadcopter13D_HJI(HJI_Solver):
    """
    13D Quadcopter HJI Reachability Problem.
    
    State: x = (p_x, p_y, p_z, q_w, q_x, q_y, q_z, v_x, v_y, v_z, omega_x, omega_y, omega_z)
    where:
        - (p_x, p_y, p_z) ∈ [-3, 3]^3: position
        - (q_w, q_x, q_y, q_z) ∈ [-1, 1]^4: quaternion orientation
        - (v_x, v_y, v_z) ∈ [-5, 5]^3: linear velocities
        - (omega_x, omega_y, omega_z) ∈ [-5, 5]^3: angular velocities
    
    Control: u = (F, alpha_x, alpha_y, alpha_z)
        - F ∈ [-20, 20]: collective thrust
        - alpha_x, alpha_y ∈ [-8, 8]: angular accelerations (roll, pitch)
        - alpha_z ∈ [-4, 4]: angular acceleration (yaw)
    
    Dynamics:
        p_dot = v
        q_w_dot = -0.5 * (omega_x*q_x + omega_y*q_y + omega_z*q_z)
        q_x_dot =  0.5 * (omega_x*q_w + omega_z*q_y - omega_y*q_z)
        q_y_dot =  0.5 * (omega_y*q_w - omega_z*q_x + omega_x*q_z)
        q_z_dot =  0.5 * (omega_z*q_w + omega_y*q_x - omega_x*q_y)
        v_x_dot = CT * (2*q_w*q_y + 2*q_x*q_z) * F / m
        v_y_dot = CT * (-2*q_w*q_x + 2*q_y*q_z) * F / m
        v_z_dot = Gz - CT * (2*q_x^2 + 2*q_y^2 - 1) * F / m
        omega_x_dot = alpha_x - (5/9) * omega_y * omega_z
        omega_y_dot = alpha_y + (5/9) * omega_x * omega_z
        omega_z_dot = alpha_z
    
    Safety function: distance from drone disk to cylinder obstacle
        l(x) = max(sqrt(p_x^2 + p_y^2) - sqrt(d_x + d_y), 0) - r_0
    where d_x, d_y account for disk orientation based on
    nu = q * e3 * q_bar (body z-axis in world frame).
    
    Hamiltonian (maximized over control):
        H(x, p) = C(x, p) + |A_F(x, p)| * F_max 
                  + |p_omega_x| * alpha_x_max + |p_omega_y| * alpha_y_max + |p_omega_z| * alpha_z_max
    """
    
    def __init__(self, config: Config):
        super().__init__(config)
        # Physical parameters
        self.CT = getattr(config, 'CT', 1.0)        # Lifting coefficient
        self.m = getattr(config, 'm', 1.0)          # Mass
        self.Gz = getattr(config, 'Gz', -9.81)      # Gravity (negative = downward)
        
        # Control bounds
        self.F_max = getattr(config, 'F_max', 20.0)
        self.F_min = getattr(config, 'F_min', -20.0)
        self.alpha_x_max = getattr(config, 'alpha_x_max', 8.0)
        self.alpha_x_min = getattr(config, 'alpha_x_min', -8.0)
        self.alpha_y_max = getattr(config, 'alpha_y_max', 8.0)
        self.alpha_y_min = getattr(config, 'alpha_y_min', -8.0)
        self.alpha_z_max = getattr(config, 'alpha_z_max', 4.0)
        self.alpha_z_min = getattr(config, 'alpha_z_min', -4.0)
        
        # Safety parameters
        self.r_a = getattr(config, 'r_a', 0.17)     # Drone radius
        self.r_0 = getattr(config, 'r_0', 0.5)      # Cylinder obstacle radius
        
        # Small epsilon for numerical stability
        self.eps = 1e-6
    
    def get_control_bounds(self):
        """
        Control bounds for 13D quadcopter: u = [F, alpha_x, alpha_y, alpha_z]
        Returns arrays for per-dimension bounds.
        """
        u_min = jnp.array([self.F_min, self.alpha_x_min, self.alpha_y_min, self.alpha_z_min])
        u_max = jnp.array([self.F_max, self.alpha_x_max, self.alpha_y_max, self.alpha_z_max])
        return u_min, u_max, 4
    
    def f(self, x, u, d):
        """
        13D Quadcopter dynamics: dx/dt = f(x, u)
        
        Args:
            x: State [p_x, p_y, p_z, q_w, q_x, q_y, q_z, v_x, v_y, v_z, omega_x, omega_y, omega_z]
               shape (batch, 13)
            u: Control [F, alpha_x, alpha_y, alpha_z], shape (batch, 4)
            d: Disturbance (unused for this problem), shape (batch, 1) or (batch, 4)
        
        Returns:
            dx/dt: shape (batch, 13)
        """
        def dynamics_single(x_i, u_i, d_i):
            # Unpack state
            p_x, p_y, p_z = x_i[0], x_i[1], x_i[2]
            q_w, q_x, q_y, q_z = x_i[3], x_i[4], x_i[5], x_i[6]
            v_x, v_y, v_z = x_i[7], x_i[8], x_i[9]
            omega_x, omega_y, omega_z = x_i[10], x_i[11], x_i[12]
            
            # Unpack control
            F = u_i[0]
            alpha_x, alpha_y, alpha_z = u_i[1], u_i[2], u_i[3]
            
            # Position dynamics: p_dot = v
            p_x_dot = v_x
            p_y_dot = v_y
            p_z_dot = v_z
            
            # Quaternion dynamics
            q_w_dot = -0.5 * (omega_x * q_x + omega_y * q_y + omega_z * q_z)
            q_x_dot =  0.5 * (omega_x * q_w + omega_z * q_y - omega_y * q_z)
            q_y_dot =  0.5 * (omega_y * q_w - omega_z * q_x + omega_x * q_z)
            q_z_dot =  0.5 * (omega_z * q_w + omega_y * q_x - omega_x * q_y)
            
            # Linear velocity dynamics
            v_x_dot = self.CT * (2 * q_w * q_y + 2 * q_x * q_z) * F / self.m
            v_y_dot = self.CT * (-2 * q_w * q_x + 2 * q_y * q_z) * F / self.m
            v_z_dot = self.Gz - self.CT * (2 * q_x**2 + 2 * q_y**2 - 1) * F / self.m
            
            # Angular velocity dynamics
            omega_x_dot = alpha_x - (5.0 / 9.0) * omega_y * omega_z
            omega_y_dot = alpha_y + (5.0 / 9.0) * omega_x * omega_z
            omega_z_dot = alpha_z
            
            return jnp.array([
                p_x_dot, p_y_dot, p_z_dot,
                q_w_dot, q_x_dot, q_y_dot, q_z_dot,
                v_x_dot, v_y_dot, v_z_dot,
                omega_x_dot, omega_y_dot, omega_z_dot
            ])
        
        return jax.vmap(dynamics_single, in_axes=(0, 0, 0))(x, u, d)
    
    def l(self, x):
        """
        Safety function (batched): distance from drone disk to cylinder obstacle.
        
        l(x) = max(sqrt(p_x^2 + p_y^2) - sqrt(d_x + d_y), 0) - r_0
        
        where d_x, d_y account for the projection of the tilted disk onto the xy-plane.
        
        Args:
            x: State, shape (batch, 13)
        
        Returns:
            Signed distance, shape (batch, 1)
        """
        p_x = x[..., 0]
        p_y = x[..., 1]
        v_x = x[..., 7]
        v_y = x[..., 8]
        v_z = x[..., 9]

        # Horizontal distance from z-axis
        r_xy = jnp.sqrt(p_x**2 + p_y**2 + self.eps)

        # Velocity-based projection terms (reverted behavior)
        denom = (p_x**2 * v_x**2 + p_x**2 * v_z**2 + 2 * p_x * p_y * v_x * v_y
                 + p_y**2 * v_y**2 + p_y**2 * v_z**2 + self.eps)

        d_x = (self.r_a**2 * p_x**2 * v_z**2) / denom
        d_y = (self.r_a**2 * p_y**2 * v_z**2) / denom

        disk_radius = jnp.sqrt(d_x + d_y + self.eps)
        l_val = jnp.maximum(r_xy - disk_radius, 0.0) - self.r_0
        return l_val[..., jnp.newaxis]

    
    def _C(self, x, dv_flat):
        """
        Control-independent part of the Hamiltonian: C(x, p).
        
        C(x,p) = p_px * v_x + p_py * v_y + p_pz * v_z
               + p_qw * (-0.5*(omega_x*q_x + omega_y*q_y + omega_z*q_z))
               + p_qx * ( 0.5*(omega_x*q_w + omega_z*q_y - omega_y*q_z))
               + p_qy * ( 0.5*(omega_y*q_w - omega_z*q_x + omega_x*q_z))
               + p_qz * ( 0.5*(omega_z*q_w + omega_y*q_x - omega_x*q_y))
               + p_vz * Gz
               + p_omega_x * (-5/9 * omega_y * omega_z)
               + p_omega_y * ( 5/9 * omega_x * omega_z)
        
        Args:
            x: State, shape (batch, 13)
            dv_flat: Gradient of value function, shape (batch, 13)
        
        Returns:
            C values, shape (batch,)
        """
        # Unpack state
        q_w, q_x, q_y, q_z = x[:, 3], x[:, 4], x[:, 5], x[:, 6]
        v_x, v_y, v_z = x[:, 7], x[:, 8], x[:, 9]
        omega_x, omega_y, omega_z = x[:, 10], x[:, 11], x[:, 12]
        
        # Unpack costate (gradient)
        p_px, p_py, p_pz = dv_flat[:, 0], dv_flat[:, 1], dv_flat[:, 2]
        p_qw, p_qx, p_qy, p_qz = dv_flat[:, 3], dv_flat[:, 4], dv_flat[:, 5], dv_flat[:, 6]
        p_vx, p_vy, p_vz = dv_flat[:, 7], dv_flat[:, 8], dv_flat[:, 9]
        p_omega_x, p_omega_y, p_omega_z = dv_flat[:, 10], dv_flat[:, 11], dv_flat[:, 12]
        
        # Position velocity terms
        C = p_px * v_x + p_py * v_y + p_pz * v_z
        
        # Quaternion dynamics terms
        C = C + p_qw * (-0.5 * (omega_x * q_x + omega_y * q_y + omega_z * q_z))
        C = C + p_qx * ( 0.5 * (omega_x * q_w + omega_z * q_y - omega_y * q_z))
        C = C + p_qy * ( 0.5 * (omega_y * q_w - omega_z * q_x + omega_x * q_z))
        C = C + p_qz * ( 0.5 * (omega_z * q_w + omega_y * q_x - omega_x * q_y))
        
        # Gravity term
        C = C + p_vz * self.Gz
        
        # Angular velocity coupling terms
        C = C + p_omega_x * (-5.0 / 9.0 * omega_y * omega_z)
        C = C + p_omega_y * ( 5.0 / 9.0 * omega_x * omega_z)
        
        return C
    
    def _A_F(self, x, dv_flat):
        """
        Switching function for thrust F.
        
        A_F(x, p) = p_vx * (2*q_w*q_y + 2*q_x*q_z)
                  + p_vy * (-2*q_w*q_x + 2*q_y*q_z)
                  - p_vz * (2*q_x^2 + 2*q_y^2 - 1)
        
        Note: The coefficient of F in dynamics is CT/m * rotation_term, so:
        A_F = (CT/m) * [p_vx*(2qw*qy + 2qx*qz) + p_vy*(-2qw*qx + 2qy*qz) - p_vz*(2qx^2 + 2qy^2 - 1)]
        
        Args:
            x: State, shape (batch, 13)
            dv_flat: Gradient of value function, shape (batch, 13)
        
        Returns:
            A_F values, shape (batch,)
        """
        q_w, q_x, q_y, q_z = x[:, 3], x[:, 4], x[:, 5], x[:, 6]
        p_vx, p_vy, p_vz = dv_flat[:, 7], dv_flat[:, 8], dv_flat[:, 9]
        
        A_F = (self.CT / self.m) * (
            p_vx * (2 * q_w * q_y + 2 * q_x * q_z)
            + p_vy * (-2 * q_w * q_x + 2 * q_y * q_z)
            - p_vz * (2 * q_x**2 + 2 * q_y**2 - 1)
        )
        
        return A_F
    
    def u_star(self, x, dv, *, smooth: bool = False):
        """
        Optimal control (maximizer of Hamiltonian).
        
        Bang-bang control:
            F* = F_max if A_F > 0, else F_min
            alpha_x* = alpha_x_max if p_omega_x > 0, else alpha_x_min
            alpha_y* = alpha_y_max if p_omega_y > 0, else alpha_y_min
            alpha_z* = alpha_z_max if p_omega_z > 0, else alpha_z_min
        
        Args:
            x: State, shape (batch, 13)
            dv: Gradient of value function, shape (batch, d_out, d_in) = (batch, 1, 13)
        
        Returns:
            Optimal control [F, alpha_x, alpha_y, alpha_z], shape (batch, 4)
        """
        dv_flat = dv[:, 0, :]  # (batch, 13)
        
        # Thrust switching function
        A_F = self._A_F(x, dv_flat)
        
        # Angular acceleration costates
        p_omega_x = dv_flat[:, 10]
        p_omega_y = dv_flat[:, 11]
        p_omega_z = dv_flat[:, 12]

        if smooth:
            s_F = self.smooth_sign(A_F)
            F_star = self.map_to_bounds(s_F, self.F_min, self.F_max)

            s_x = self.smooth_sign(p_omega_x)
            s_y = self.smooth_sign(p_omega_y)
            s_z = self.smooth_sign(p_omega_z)
            alpha_x_star = self.map_to_bounds(s_x, self.alpha_x_min, self.alpha_x_max)
            alpha_y_star = self.map_to_bounds(s_y, self.alpha_y_min, self.alpha_y_max)
            alpha_z_star = self.map_to_bounds(s_z, self.alpha_z_min, self.alpha_z_max)
        else:
            F_star = jnp.where(A_F >= 0, self.F_max, self.F_min)
            alpha_x_star = jnp.where(p_omega_x >= 0, self.alpha_x_max, self.alpha_x_min)
            alpha_y_star = jnp.where(p_omega_y >= 0, self.alpha_y_max, self.alpha_y_min)
            alpha_z_star = jnp.where(p_omega_z >= 0, self.alpha_z_max, self.alpha_z_min)
        
        return jnp.stack([F_star, alpha_x_star, alpha_y_star, alpha_z_star], axis=-1)
    
    def d_star(self, x, dv, *, smooth: bool = False):
        """
        Optimal disturbance (minimizer).
        For this problem, there is no disturbance, so return zeros.
        
        Args:
            x: State, shape (batch, 13)
            dv: Gradient of value function, shape (batch, 1, 13)
        
        Returns:
            Zero disturbance, shape (batch, 4) to match control dimension
        """
        return jnp.zeros((x.shape[0], 4))
    
    def H(self, x, dv):
        """
        Analytical Hamiltonian under optimal control (maximized).
        
        H(x, p) = C(x, p) + |A_F(x, p)| * F_max 
                  + |p_omega_x| * alpha_x_max 
                  + |p_omega_y| * alpha_y_max 
                  + |p_omega_z| * alpha_z_max
        
        (Using symmetric bounds: F_min = -F_max, alpha_i_min = -alpha_i_max)
        
        Args:
            x: State, shape (batch, 13)
            dv: Gradient of value function, shape (batch, 1, 13)
        
        Returns:
            Hamiltonian values, shape (batch, 1)
        """
        if self._use_smooth_controls(training=False):
            return super().H(x, dv)

        dv_flat = dv[:, 0, :]  # (batch, 13)
        
        # Control-independent part
        C = self._C(x, dv_flat)
        
        # Thrust contribution (symmetric bounds)
        A_F = self._A_F(x, dv_flat)
        H_F = jnp.abs(A_F) * self.F_max
        
        # Angular acceleration contributions (symmetric bounds)
        p_omega_x = dv_flat[:, 10]
        p_omega_y = dv_flat[:, 11]
        p_omega_z = dv_flat[:, 12]
        
        H_alpha = (jnp.abs(p_omega_x) * self.alpha_x_max 
                   + jnp.abs(p_omega_y) * self.alpha_y_max 
                   + jnp.abs(p_omega_z) * self.alpha_z_max)
        
        H_val = C + H_F + H_alpha
        
        return H_val[:, jnp.newaxis]
    
    def sample_domain(self, key: Key, batch_size):
        """Sample from 13D state space for training."""
        # Position: p_x, p_y, p_z ∈ [-3, 3]
        p_x = jax.random.uniform(key.newkey(), (batch_size, 1), minval=-3.0, maxval=3.0)
        p_y = jax.random.uniform(key.newkey(), (batch_size, 1), minval=-3.0, maxval=3.0)
        p_z = jax.random.uniform(key.newkey(), (batch_size, 1), minval=-3.0, maxval=3.0)
        
        # Quaternion: q_w, q_x, q_y, q_z ∈ [-1, 1]
        q_w = jax.random.uniform(key.newkey(), (batch_size, 1), minval=-1.0, maxval=1.0)
        q_x = jax.random.uniform(key.newkey(), (batch_size, 1), minval=-1.0, maxval=1.0)
        q_y = jax.random.uniform(key.newkey(), (batch_size, 1), minval=-1.0, maxval=1.0)
        q_z = jax.random.uniform(key.newkey(), (batch_size, 1), minval=-1.0, maxval=1.0)
        
        # Optionally normalize quaternion for valid rotation
        q_norm = jnp.sqrt(q_w**2 + q_x**2 + q_y**2 + q_z**2 + self.eps)
        q_w, q_x, q_y, q_z = q_w/q_norm, q_x/q_norm, q_y/q_norm, q_z/q_norm
        
        # Linear velocity: v_x, v_y, v_z ∈ [-5, 5]
        v_x = jax.random.uniform(key.newkey(), (batch_size, 1), minval=-5.0, maxval=5.0)
        v_y = jax.random.uniform(key.newkey(), (batch_size, 1), minval=-5.0, maxval=5.0)
        v_z = jax.random.uniform(key.newkey(), (batch_size, 1), minval=-5.0, maxval=5.0)
        
        # Angular velocity: omega_x, omega_y, omega_z ∈ [-5, 5]
        omega_x = jax.random.uniform(key.newkey(), (batch_size, 1), minval=-5.0, maxval=5.0)
        omega_y = jax.random.uniform(key.newkey(), (batch_size, 1), minval=-5.0, maxval=5.0)
        omega_z = jax.random.uniform(key.newkey(), (batch_size, 1), minval=-5.0, maxval=5.0)
        
        x_pde = jnp.concatenate([
            p_x, p_y, p_z,
            q_w, q_x, q_y, q_z,
            v_x, v_y, v_z,
            omega_x, omega_y, omega_z
        ], axis=-1)
        
        t_pde = jax.random.uniform(key.newkey(), (batch_size, 1),
                                   minval=self.config.t_range[0],
                                   maxval=self.config.t_range[1])
        return x_pde, t_pde
    
    @staticmethod
    def get_base_config(traj_len=50):
        """Get configuration for 13D quadcopter problem."""
        T = 1.0
        config = Config(
            case='quadcopter_13d',
            d_in=13,
            d_out=1,
            traj_len=traj_len,
            delta_t=T / traj_len,
            x_range=(
                (-3.0, 3.0),   # p_x
                (-3.0, 3.0),   # p_y
                (-3.0, 3.0),   # p_z
                (-1.0, 1.0),   # q_w
                (-1.0, 1.0),   # q_x
                (-1.0, 1.0),   # q_y
                (-1.0, 1.0),   # q_z
                (-5.0, 5.0),   # v_x
                (-5.0, 5.0),   # v_y
                (-5.0, 5.0),   # v_z
                (-5.0, 5.0),   # omega_x
                (-5.0, 5.0),   # omega_y
                (-5.0, 5.0),   # omega_z
            ),
            t_range=(0, T),
            periodic=False,
            random_sample=True,
        )
        # Physical parameters
        config.CT = 1.0          # Lifting coefficient
        config.m = 1.0           # Mass
        config.Gz = -9.81        # Gravity (downward)
        
        # Control bounds
        config.F_max = 20.0
        config.F_min = -20.0
        config.alpha_x_max = 8.0
        config.alpha_x_min = -8.0
        config.alpha_y_max = 8.0
        config.alpha_y_min = -8.0
        config.alpha_z_max = 4.0
        config.alpha_z_min = -4.0
        
        # Safety parameters
        config.r_a = 0.17        # Drone radius
        config.r_0 = 0.5         # Cylinder obstacle radius
        
        # Noise for BSDE
        config.sigma_noise = 0.01
        
        return config


class F1tenth_HJI(HJI_Solver):
    """
    DeepReach-faithful F1Tenth avoid-style HJI problem.

    State order:
        x = [x_pos, y_pos, delta, v, theta, omega, beta]
    Control:
        u = [steering_rate, acceleration]
    """

    def __init__(self, config: Config):
        super().__init__(config)

        # Vehicle parameters (matching DeepReach defaults).
        self.mu = float(getattr(config, "mu", 1.0489))
        self.C_Sf = float(getattr(config, "C_Sf", 4.718))
        self.C_Sr = float(getattr(config, "C_Sr", 5.4562))
        self.lf = float(getattr(config, "lf", 0.15875))
        self.lr = float(getattr(config, "lr", 0.17145))
        self.h = float(getattr(config, "h", 0.074))
        self.m = float(getattr(config, "m", 3.74))
        self.I = float(getattr(config, "I", 0.04712))
        self.g = float(getattr(config, "g", 9.81))
        self.lwb = self.lf + self.lr

        # State/control constraints.
        self.s_min = float(getattr(config, "s_min", -0.4189))
        self.s_max = float(getattr(config, "s_max", 0.4189))
        self.sv_min = float(getattr(config, "sv_min", -3.2))
        self.sv_max = float(getattr(config, "sv_max", 3.2))
        self.v_switch = float(getattr(config, "v_switch", 7.319))
        self.a_max = float(getattr(config, "a_max", 9.51))
        self.v_min = float(getattr(config, "v_min", 0.1))
        self.v_max = float(getattr(config, "v_max", 10.0))
        self.omega_max = float(getattr(config, "omega_max", 6.0))

        # Map/world coordinates.
        self.pixel2world = float(getattr(config, "pixel2world", 0.0625))
        self.x_min = float(getattr(config, "x_min", 0.0))
        self.x_max = float(getattr(config, "x_max", 62.5))
        self.y_min = float(getattr(config, "y_min", 0.0))
        self.y_max = float(getattr(config, "y_max", 50.0))
        self._f1_map_path = str(getattr(config, "f1_map_path", "src/assets/f1tenth/F1_map_obstaclemap.mat"))
        self.f1_use_rejection_sampling = bool(getattr(config, "f1_use_rejection_sampling", True))
        self.f1_rejection_train_l_min = float(getattr(config, "f1_rejection_train_l_min", -1.0))
        self.f1_rejection_eval_l_min = float(getattr(config, "f1_rejection_eval_l_min", 0.0))
        self.f1_rejection_oversample_factor = float(getattr(config, "f1_rejection_oversample_factor", 2.0))
        self.f1_rejection_max_rounds = int(getattr(config, "f1_rejection_max_rounds", 64))

        # Numeric stability constants.
        self._speed_eps = 1e-6
        self._cos_sq_eps = 1e-6
        self._kinematic_speed_threshold = 0.5

        if self.f1_rejection_oversample_factor < 1.0:
            raise ValueError(
                f"f1_rejection_oversample_factor must be >= 1.0, got {self.f1_rejection_oversample_factor}"
            )
        if self.f1_rejection_max_rounds < 1:
            raise ValueError(f"f1_rejection_max_rounds must be >= 1, got {self.f1_rejection_max_rounds}")

        self._sample_dtype = jnp.asarray(0.0).dtype
        x_lower = []
        x_upper = []
        for i in range(self.config.d_in):
            x_range = self.config.x_range[i] if i < len(self.config.x_range) else self.config.x_range[-1]
            x_lower.append(float(x_range[0]))
            x_upper.append(float(x_range[1]))
        self._x_lower = jnp.asarray(x_lower, dtype=self._sample_dtype)
        self._x_upper = jnp.asarray(x_upper, dtype=self._sample_dtype)
        self._t_min = jnp.asarray(self.config.t_range[0], dtype=self._sample_dtype)
        self._t_max = jnp.asarray(self.config.t_range[1], dtype=self._sample_dtype)

        self._load_obstacle_map()

    def _resolve_f1_map_path(self) -> Path:
        raw = Path(self._f1_map_path).expanduser()
        if raw.is_absolute():
            candidates = [raw]
        else:
            src_dir = Path(__file__).resolve().parent
            repo_root = src_dir.parent
            candidates = [
                Path.cwd() / raw,
                repo_root / raw,
                src_dir / raw,
                src_dir / "assets" / "f1tenth" / raw.name,
            ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        tried = "\n".join(str(c) for c in candidates)
        raise FileNotFoundError(
            "F1 obstacle map not found. Set config.f1_map_path to a valid .mat file.\n"
            f"Tried:\n{tried}"
        )

    def _load_obstacle_map(self) -> None:
        try:
            from scipy.io import loadmat
        except Exception as exc:  # pragma: no cover - environment-dependent import
            raise ImportError(
                "scipy is required to load the F1Tenth obstacle map (.mat). "
                "Install scipy or provide a pre-loaded alternative."
            ) from exc

        map_path = self._resolve_f1_map_path()
        mat = loadmat(map_path)
        if "obs_map" not in mat:
            raise KeyError(f"obs_map key not found in F1 map file: {map_path}")

        obstacle_map = np.asarray(mat["obs_map"], dtype=np.float64)
        obstacle_map = obstacle_map.copy()
        # Match DeepReach preprocessing: convert exact zeros (including -0.0) to +1.
        obstacle_map[obstacle_map == 0.0] = 1.0

        row_start = int(self.y_min / self.pixel2world)
        row_end = int(self.y_max / self.pixel2world) + 1
        col_start = int(self.x_min / self.pixel2world)
        col_end = int(self.x_max / self.pixel2world) + 1
        obstacle_map = obstacle_map[row_start:row_end, col_start:col_end]

        if obstacle_map.ndim != 2 or obstacle_map.size == 0:
            raise ValueError(
                f"Loaded F1 obstacle map has invalid shape {obstacle_map.shape} after cropping."
            )

        self.obstacle_map = jnp.asarray(obstacle_map)
        self.map_height = int(obstacle_map.shape[0])
        self.map_width = int(obstacle_map.shape[1])
        self.f1_map_path_resolved = str(map_path)

    def _bilinear_interpolate_map(self, row_coords, col_coords):
        row = jnp.asarray(row_coords, dtype=self.obstacle_map.dtype)
        col = jnp.asarray(col_coords, dtype=self.obstacle_map.dtype)

        row0 = jnp.floor(row).astype(jnp.int32)
        col0 = jnp.floor(col).astype(jnp.int32)
        row1 = row0 + 1
        col1 = col0 + 1

        row0 = jnp.clip(row0, 0, self.map_height - 1)
        row1 = jnp.clip(row1, 0, self.map_height - 1)
        col0 = jnp.clip(col0, 0, self.map_width - 1)
        col1 = jnp.clip(col1, 0, self.map_width - 1)

        v00 = self.obstacle_map[row0, col0]
        v01 = self.obstacle_map[row0, col1]
        v10 = self.obstacle_map[row1, col0]
        v11 = self.obstacle_map[row1, col1]

        row_frac = row - row0.astype(self.obstacle_map.dtype)
        col_frac = col - col0.astype(self.obstacle_map.dtype)

        v0 = v00 * (1.0 - row_frac) + v10 * row_frac
        v1 = v01 * (1.0 - row_frac) + v11 * row_frac
        return v0 * (1.0 - col_frac) + v1 * col_frac

    def _sample_uniform_domain_with_raw_key(self, raw_key, num_samples: int):
        raw_key, x_key, t_key = jax.random.split(raw_key, 3)
        x_unit = jax.random.uniform(
            x_key,
            (num_samples, self.config.d_in),
            minval=0.0,
            maxval=1.0,
            dtype=self._sample_dtype,
        )
        x = self._x_lower + (self._x_upper - self._x_lower) * x_unit
        t = jax.random.uniform(
            t_key,
            (num_samples, 1),
            minval=self._t_min,
            maxval=self._t_max,
            dtype=self._sample_dtype,
        )
        return raw_key, x, t

    def _sample_domain_rejection(self, key: Key, num_samples: int, l_min: float):
        if num_samples <= 0:
            x_empty = jnp.zeros((0, self.config.d_in))
            t_empty = jnp.zeros((0, 1))
            return x_empty, t_empty

        l_threshold = jnp.asarray(float(l_min), dtype=self._sample_dtype)
        proposal_n = max(1, int(np.ceil(num_samples * self.f1_rejection_oversample_factor)))

        x_init = jnp.zeros((num_samples, self.config.d_in), dtype=self._sample_dtype)
        t_init = jnp.zeros((num_samples, 1), dtype=self._sample_dtype)
        count_init = jnp.asarray(0, dtype=jnp.int32)
        round_init = jnp.asarray(0, dtype=jnp.int32)
        raw_key_init = key.newkey()

        def cond_fn(carry):
            round_idx, accepted_count, _, _, _ = carry
            return jnp.logical_and(round_idx < self.f1_rejection_max_rounds, accepted_count < num_samples)

        def body_fn(carry):
            round_idx, accepted_count, x_acc, t_acc, raw_key = carry
            raw_key, x_prop, t_prop = self._sample_uniform_domain_with_raw_key(raw_key, proposal_n)

            l_prop = self.l(x_prop)[:, 0]
            accept_mask = l_prop >= l_threshold
            accept_int = accept_mask.astype(jnp.int32)
            accept_prefix = jnp.cumsum(accept_int)

            remaining = jnp.maximum(num_samples - accepted_count, 0)
            take_mask = jnp.logical_and(accept_mask, accept_prefix <= remaining)
            take_int = take_mask.astype(jnp.int32)

            local_idx = jnp.cumsum(take_int) - 1
            global_idx = accepted_count + local_idx
            safe_idx = jnp.clip(global_idx, 0, num_samples - 1)

            x_updates = x_prop * take_mask[:, jnp.newaxis].astype(x_prop.dtype)
            t_updates = t_prop * take_mask[:, jnp.newaxis].astype(t_prop.dtype)
            x_acc = x_acc.at[safe_idx].add(x_updates)
            t_acc = t_acc.at[safe_idx].add(t_updates)

            accepted_in_round = jnp.sum(take_int, dtype=accepted_count.dtype).astype(accepted_count.dtype)
            accepted_count = accepted_count + accepted_in_round
            next_round_idx = round_idx + jnp.asarray(1, dtype=round_idx.dtype)
            return next_round_idx, accepted_count, x_acc, t_acc, raw_key

        _, accepted_count, x_out, t_out, _ = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (round_init, count_init, x_init, t_init, raw_key_init),
        )

        error_message = (
            "F1 rejection sampling failed to collect enough points. "
            f"Required {num_samples} accepted points after {self.f1_rejection_max_rounds} rounds "
            f"with l(x) >= {float(l_min)}. Adjust thresholds or rejection settings."
        )

        def _raise_if_insufficient(count):
            count_int = int(np.asarray(count))
            if count_int < num_samples:
                raise RuntimeError(error_message)

        if isinstance(accepted_count, jax.core.Tracer):
            jax.debug.callback(_raise_if_insufficient, accepted_count)
        else:
            _raise_if_insufficient(accepted_count)

        return x_out, t_out

    def sample_domain(self, key: Key, num_samples):
        if not self.f1_use_rejection_sampling:
            return HJI_Solver.sample_domain(self, key, num_samples)
        return self._sample_domain_rejection(key, int(num_samples), self.f1_rejection_train_l_min)

    def sample_domain_eval(self, key: Key, num_samples):
        if not self.f1_use_rejection_sampling:
            return HJI_Solver.sample_domain(self, key, num_samples)
        return self._sample_domain_rejection(key, int(num_samples), self.f1_rejection_eval_l_min)

    def clamp_control(self, x, u):
        """State-dependent control clamping matching DeepReach F1tenth."""
        x = jnp.asarray(x)
        u = jnp.asarray(u)

        steer_rate = jnp.clip(u[..., 0], self.sv_min, self.sv_max)
        accel = jnp.clip(u[..., 1], -self.a_max, self.a_max)

        delta = x[..., 2]
        steer_rate = jnp.where(delta > (self.s_max - 0.01), jnp.minimum(steer_rate, 0.0), steer_rate)
        steer_rate = jnp.where(delta < (self.s_min + 0.01), jnp.maximum(steer_rate, 0.0), steer_rate)

        v = x[..., 3]
        v_safe = jnp.maximum(v, self._speed_eps)
        accel_upper = jnp.where(v > self.v_switch, self.a_max * self.v_switch / v_safe, self.a_max)
        accel = jnp.minimum(accel, accel_upper)

        return jnp.stack([steer_rate, accel], axis=-1)

    def get_control_bounds(self):
        u_min = jnp.array([self.sv_min, -self.a_max])
        u_max = jnp.array([self.sv_max, self.a_max])
        return u_min, u_max, 2

    def f(self, x, u, d):
        """
        Hybrid F1Tenth dynamics with low-speed kinematic and high-speed dynamic models.
        """
        x = jnp.asarray(x)
        u = self.clamp_control(x, jnp.asarray(u))

        delta = x[..., 2]
        v = x[..., 3]
        theta = x[..., 4]
        omega = x[..., 5]
        beta = x[..., 6]

        sv = u[..., 0]
        acc = u[..., 1]

        cos_delta = jnp.cos(delta)
        cos_delta_sq = jnp.maximum(cos_delta**2, self._cos_sq_eps)
        tan_delta = jnp.tan(delta)

        v_safe = jnp.where(
            jnp.abs(v) < self._kinematic_speed_threshold,
            jnp.where(v >= 0.0, self._kinematic_speed_threshold, -self._kinematic_speed_threshold),
            v,
        )
        v_sq_safe = jnp.maximum(v_safe**2, self._speed_eps)

        # Kinematic branch (|v| < threshold).
        f_kin_0 = v * jnp.cos(theta)
        f_kin_1 = v * jnp.sin(theta)
        f_kin_2 = sv
        f_kin_3 = acc
        f_kin_4 = (v / self.lwb) * tan_delta
        f_kin_5 = (acc / self.lwb) * tan_delta + (v / self.lwb) * (sv / cos_delta_sq)
        f_kin_6 = jnp.zeros_like(v)

        # Dynamic branch.
        g_lr_minus_ah = self.g * self.lr - acc * self.h
        g_lf_plus_ah = self.g * self.lf + acc * self.h
        f_dyn_0 = v * jnp.cos(beta + theta)
        f_dyn_1 = v * jnp.sin(beta + theta)
        f_dyn_2 = sv
        f_dyn_3 = acc
        f_dyn_4 = omega
        f_dyn_5 = (
            -self.mu
            * self.m
            / (v_safe * self.I * (self.lr + self.lf))
            * (self.lf**2 * self.C_Sf * g_lr_minus_ah + self.lr**2 * self.C_Sr * g_lf_plus_ah)
            * omega
            + self.mu
            * self.m
            / (self.I * (self.lr + self.lf))
            * (self.lr * self.C_Sr * g_lf_plus_ah - self.lf * self.C_Sf * g_lr_minus_ah)
            * beta
            + self.mu
            * self.m
            / (self.I * (self.lr + self.lf))
            * self.lf
            * self.C_Sf
            * g_lr_minus_ah
            * delta
        )
        f_dyn_6 = (
            (
                self.mu
                / (v_sq_safe * (self.lr + self.lf))
                * (self.C_Sr * g_lf_plus_ah * self.lr - self.C_Sf * g_lr_minus_ah * self.lf)
                - 1.0
            )
            * omega
            - self.mu
            / (v_safe * (self.lr + self.lf))
            * (self.C_Sr * g_lf_plus_ah + self.C_Sf * g_lr_minus_ah)
            * beta
            + self.mu
            / (v_safe * (self.lr + self.lf))
            * (self.C_Sf * g_lr_minus_ah)
            * delta
        )

        kinematic_mask = jnp.abs(v) < self._kinematic_speed_threshold
        return jnp.stack(
            [
                jnp.where(kinematic_mask, f_kin_0, f_dyn_0),
                jnp.where(kinematic_mask, f_kin_1, f_dyn_1),
                jnp.where(kinematic_mask, f_kin_2, f_dyn_2),
                jnp.where(kinematic_mask, f_kin_3, f_dyn_3),
                jnp.where(kinematic_mask, f_kin_4, f_dyn_4),
                jnp.where(kinematic_mask, f_kin_5, f_dyn_5),
                jnp.where(kinematic_mask, f_kin_6, f_dyn_6),
            ],
            axis=-1,
        )

    def l(self, x):
        """
        Obstacle-map signed field from bilinear interpolation.
        Positive values are safe, negative values are unsafe.
        """
        x = jnp.asarray(x)
        x_pos = x[..., 0]
        y_pos = x[..., 1]

        # DeepReach uses image-space query [row=y, col=x].
        row = (y_pos - self.y_min) / self.pixel2world
        col = (x_pos - self.x_min) / self.pixel2world
        obstacle_value = self._bilinear_interpolate_map(row, col)
        return obstacle_value[..., jnp.newaxis]

    def u_star(self, x, dv, *, smooth: bool = False):
        """
        Optimal avoid control matching DeepReach F1tenth switching surfaces.
        """
        x = jnp.asarray(x)
        dv_flat = dv[:, 0, :]

        delta = x[:, 2]
        v = x[:, 3]
        omega = x[:, 5]
        beta = x[:, 6]

        p_delta = dv_flat[:, 2]
        p_v = dv_flat[:, 3]
        p_omega = dv_flat[:, 5]
        p_beta = dv_flat[:, 6]

        cos_delta = jnp.cos(delta)
        cos_delta_sq = jnp.maximum(cos_delta**2, self._cos_sq_eps)
        tan_delta = jnp.tan(delta)

        v_safe = jnp.where(
            jnp.abs(v) < self._kinematic_speed_threshold,
            jnp.where(v >= 0.0, self._kinematic_speed_threshold, -self._kinematic_speed_threshold),
            v,
        )
        v_sq_safe = jnp.maximum(v_safe**2, self._speed_eps)

        # Kinematic switching functions.
        switch_steer_kin = p_delta + p_omega * v / (self.lwb * cos_delta_sq)
        switch_acc_kin = p_v + p_omega * tan_delta / self.lwb

        # Dynamic switching functions.
        dyn_term_omega = (
            -self.mu
            * self.m
            / (v_safe * self.I * (self.lr + self.lf))
            * (-self.lf**2 * self.C_Sf * self.h + self.lr**2 * self.C_Sr * self.h)
            * omega
            + self.mu
            * self.m
            / (self.I * (self.lr + self.lf))
            * (self.lr * self.C_Sr * self.h + self.lf * self.C_Sf * self.h)
            * beta
            - self.mu
            * self.m
            / (self.I * (self.lr + self.lf))
            * self.lf
            * self.C_Sf
            * self.h
            * delta
        )
        dyn_term_beta = (
            self.mu
            / (v_sq_safe * (self.lr + self.lf))
            * (self.C_Sr * self.h * self.lr + self.C_Sf * self.h * self.lf)
            * omega
            - self.mu
            / (v_safe * (self.lr + self.lf))
            * (self.C_Sr * self.h - self.C_Sf * self.h)
            * beta
            - self.mu
            / (v_safe * (self.lr + self.lf))
            * self.C_Sf
            * self.h
            * delta
        )
        switch_steer_dyn = p_delta
        switch_acc_dyn = p_v + p_omega * dyn_term_omega + p_beta * dyn_term_beta

        if smooth:
            steer_kin = self.sv_max * self.smooth_sign(switch_steer_kin)
            accel_kin = self.a_max * self.smooth_sign(switch_acc_kin)
            steer_dyn = self.sv_max * self.smooth_sign(switch_steer_dyn)
            accel_dyn = self.a_max * self.smooth_sign(switch_acc_dyn)
        else:
            steer_kin = self.sv_max * jnp.sign(switch_steer_kin)
            accel_kin = self.a_max * jnp.sign(switch_acc_kin)
            steer_dyn = self.sv_max * jnp.sign(switch_steer_dyn)
            accel_dyn = self.a_max * jnp.sign(switch_acc_dyn)

        kinematic_mask = jnp.abs(v) < self._kinematic_speed_threshold
        u = jnp.stack(
            [
                jnp.where(kinematic_mask, steer_kin, steer_dyn),
                jnp.where(kinematic_mask, accel_kin, accel_dyn),
            ],
            axis=-1,
        )
        return self.clamp_control(x, u)

    def d_star(self, x, dv, *, smooth: bool = False):
        """No disturbance for this problem."""
        return jnp.zeros((x.shape[0], 1))

    @staticmethod
    def get_base_config(traj_len=50):
        T = 1.0
        config = Config(
            case="f1tenth",
            d_in=7,
            d_out=1,
            traj_len=traj_len,
            delta_t=T / traj_len,
            x_range=(
                (0.0, 62.5),       # x
                (0.0, 50.0),       # y
                (-0.4189, 0.4189), # steering angle delta
                (0.1, 10.0),       # velocity
                (-jnp.pi, jnp.pi), # heading angle theta (periodic)
                (-6.0, 6.0),       # yaw rate omega
                (-1.0, 1.0),       # slip angle beta
            ),
            t_range=(0, T),
            periodic=True,
            periodic_idx=(4,),
            random_sample=True,
        )

        # Vehicle constants.
        config.mu = 1.0489
        config.C_Sf = 4.718
        config.C_Sr = 5.4562
        config.lf = 0.15875
        config.lr = 0.17145
        config.h = 0.074
        config.m = 3.74
        config.I = 0.04712
        config.g = 9.81

        # Constraints.
        config.s_min = -0.4189
        config.s_max = 0.4189
        config.sv_min = -3.2
        config.sv_max = 3.2
        config.v_switch = 7.319
        config.a_max = 9.51
        config.v_min = 0.1
        config.v_max = 10.0
        config.omega_max = 6.0

        # Map geometry / source.
        config.pixel2world = 0.0625
        config.x_min = 0.0
        config.x_max = 62.5
        config.y_min = 0.0
        config.y_max = 50.0
        config.f1_map_path = "src/assets/f1tenth/F1_map_obstaclemap.mat"
        config.f1_use_rejection_sampling = True
        config.f1_rejection_train_l_min = -1.0
        config.f1_rejection_eval_l_min = 0.0
        config.f1_rejection_oversample_factor = 2.0
        config.f1_rejection_max_rounds = 64

        config.sigma_noise = 0.01
        config.bound_rollout_states = True
        return config


class PubSubND_HJI(HJI_Solver):
    """
    N-dimensional Publisher-Subscriber HJI Reachability Problem.

    State: x = (x_0, x_1, ..., x_{d_in-1}), where x_0 is the publisher and
    subscribers are x_1..x_{d_in-1}. For this family, n_subs = d_in - 1.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        if int(config.d_in) < 2:
            raise ValueError(f"PubSubND_HJI requires d_in >= 2, got {config.d_in}")

        inferred_n_subs = int(config.d_in) - 1
        provided_n_subs = int(getattr(config, "n_subs", inferred_n_subs))
        if provided_n_subs != inferred_n_subs:
            raise ValueError(
                f"PubSubND_HJI expects n_subs=d_in-1 ({inferred_n_subs}), got n_subs={provided_n_subs}"
            )

        self.gamma = getattr(config, "gamma", 20.0)      # Nonlinear coupling coefficient
        self.mu = getattr(config, "mu", 0.0)             # Publisher mean-reversion (unused)
        self.alpha = getattr(config, "alpha", 0.0)       # Publisher nonlinear term
        self.a = getattr(config, "a", -0.5)              # Linear drift coefficient
        self.b_coeff = getattr(config, "b_coeff", 0.4)   # Control gain
        self.u_max = getattr(config, "u_max", 0.5)       # Control bound
        self.n_subs = inferred_n_subs
        self.r_sq = getattr(config, "r_sq", 0.0625)      # Squared target radius
        self.config.n_subs = self.n_subs

    def f(self, x, u, d):
        """
        Pub-Sub dynamics.

        Args:
            x: State [x_0, x_1, ..., x_{d_in-1}], shape (batch, d_in)
            u: Control [u_1, ..., u_{d_in-1}], shape (batch, n_subs)
            d: Disturbance (unused), shape (batch, 1)

        Returns:
            dx/dt: shape (batch, d_in)
        """

        def dynamics(x_i, u_i, d_i):
            x0 = x_i[0]
            x_subs = x_i[1:]

            x0_dot = self.a * x0 + self.alpha * jnp.sin(x0) * x0**2
            x_subs_dot = -x0 + self.a * x_subs + self.b_coeff * u_i - self.gamma * x0**2 * x_subs
            return jnp.concatenate([x0_dot[jnp.newaxis], x_subs_dot])

        return jax.vmap(dynamics, in_axes=(0, 0, 0))(x, u, d)

    def l(self, x):
        """Level set function: 0.5 * (n_subs*x0^2 + sum_i xi^2 - n_subs*r_sq)."""
        x0 = x[..., 0:1]
        x_subs = x[..., 1:]
        return 0.5 * (
            self.n_subs * x0**2
            + jnp.sum(x_subs**2, axis=-1, keepdims=True)
            - self.n_subs * self.r_sq
        )

    def u_star(self, x, dv, *, smooth: bool = False):
        """Optimal control (Hamiltonian minimizer): u_i* = -u_max * sign(∂V/∂x_i), i>=1."""
        dv_flat = dv[:, 0, :]
        p_subs = dv_flat[:, 1:]
        if smooth:
            u_min, u_max_arr, _ = self.get_control_bounds()
            s = -self.smooth_sign(p_subs)
            return self.map_to_bounds(s, u_min, u_max_arr)
        return -self.u_max * jnp.sign(p_subs)

    def d_star(self, x, dv, *, smooth: bool = False):
        """No disturbance for this problem."""
        return jnp.zeros((x.shape[0], 1))

    def H(self, x, dv):
        """
        Analytical Hamiltonian under optimal control.
        H = p0*(a*x0 + alpha*sin(x0)*x0^2)
            + sum_i p_i*(-x0 + a*x_i - gamma*x0^2*x_i)
            - b*u_max*sum_i |p_i|.
        """
        if self._use_smooth_controls(training=False):
            return super().H(x, dv)

        dv_flat = dv[:, 0, :]
        x0 = x[:, 0]
        x_subs = x[:, 1:]
        p0 = dv_flat[:, 0]
        p_subs = dv_flat[:, 1:]

        H_pub = p0 * (self.a * x0 + self.alpha * jnp.sin(x0) * x0**2)
        H_sub_drift = jnp.sum(
            p_subs * (-x0[:, jnp.newaxis] + self.a * x_subs - self.gamma * (x0**2)[:, jnp.newaxis] * x_subs),
            axis=-1,
        )
        H_control = -self.b_coeff * self.u_max * jnp.sum(jnp.abs(p_subs), axis=-1)
        return (H_pub + H_sub_drift + H_control)[:, jnp.newaxis]

    def get_control_bounds(self):
        """Control bounds: u_i in [-u_max, u_max] for all subscribers."""
        u_min = -self.u_max * jnp.ones(self.n_subs)
        u_max_arr = self.u_max * jnp.ones(self.n_subs)
        return u_min, u_max_arr, self.n_subs

    def sample_domain(self, key: Key, batch_size):
        """Sample x in [-1, 1]^d_in and t in [0, T]."""
        x_pde = jax.random.uniform(
            key.newkey(),
            (batch_size, self.config.d_in),
            minval=-1.0,
            maxval=1.0,
        )
        t_pde = jax.random.uniform(
            key.newkey(),
            (batch_size, 1),
            minval=self.config.t_range[0],
            maxval=self.config.t_range[1],
        )
        return x_pde, t_pde

    @staticmethod
    def get_base_config(traj_len=50, d_in=40, case="pubsub_nd"):
        """Get configuration for variable-dimensional PubSub."""
        d_in = int(d_in)
        if d_in < 2:
            raise ValueError(f"PubSubND_HJI requires d_in >= 2, got {d_in}")

        T = 1.0
        config = Config(
            case=case,
            d_in=d_in,
            d_out=1,
            traj_len=traj_len,
            delta_t=T / traj_len,
            x_range=((-1.0, 1.0),) * d_in,
            t_range=(0, T),
            periodic=False,
            random_sample=True,
        )
        config.gamma = 20.0
        config.mu = 0.0
        config.alpha = 0.0
        config.a = -0.5
        config.b_coeff = 0.4
        config.u_max = 0.5
        config.n_subs = d_in - 1
        config.r_sq = 0.0625
        config.sigma_noise = 0.01
        return config


class PubSub40D_HJI(PubSubND_HJI):
    """Backward-compatible 40D PubSub wrapper."""

    @staticmethod
    def get_base_config(traj_len=50):
        return PubSubND_HJI.get_base_config(traj_len=traj_len, d_in=40, case="pubsub_40d")
