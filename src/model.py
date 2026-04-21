import jax.numpy as jnp
import flax.linen as nn
import jax
import math
from config import Config

class WaveAct(nn.Module):

    @nn.compact
    def __call__(self, x):
        w1 = self.param('w1', nn.initializers.normal(.1), (x.shape[-1],))
        w2 = self.param('w2', nn.initializers.normal(.1), (x.shape[-1],))
        return jnp.asarray(w1) * jnp.sin(x) + jnp.asarray(w2) * jnp.cos(x)


class ScaledSine(nn.Module):
    omega0: float

    @nn.compact
    def __call__(self, x):
        return jnp.sin(self.omega0 * x)


class FourierEmbs(nn.Module):
    config: Config

    @nn.compact
    def __call__(self, x):
        kernel = self.param(
            "kernel", jax.nn.initializers.normal(self.config.emb_scale), (x.shape[-1], self.config.emb_dim // 2)
        )
        y = jnp.concatenate(
            [jnp.cos(jnp.dot(x, kernel)), jnp.sin(jnp.dot(x, kernel))], axis=-1
        )
        return y
  
class PINNs(nn.Module):
    config: Config

    @staticmethod
    def _fan_in(shape):
        if len(shape) == 0:
            raise ValueError("SIREN initializer requires a non-empty kernel shape.")
        if len(shape) == 1:
            return max(int(shape[0]), 1)
        return max(int(math.prod(shape[:-1])), 1)

    def _siren_first_layer_init(self):
        def init(key, shape, dtype=jnp.float32):
            fan_in = self._fan_in(shape)
            limit = 1.0 / fan_in
            return jax.random.uniform(key, shape=shape, dtype=dtype, minval=-limit, maxval=limit)
        return init

    def _siren_hidden_layer_init(self):
        omega0 = float(self.config.sine_omega0)
        if omega0 <= 0:
            raise ValueError(f"sine_omega0 must be positive, got {omega0}.")

        def init(key, shape, dtype=jnp.float32):
            fan_in = self._fan_in(shape)
            limit = jnp.sqrt(6.0 / fan_in) / omega0
            return jax.random.uniform(key, shape=shape, dtype=dtype, minval=-limit, maxval=limit)
        return init

    def _build_state_normalization_stats(self):
        centers = []
        scales = []
        d_in = int(self.config.d_in)
        x_range = tuple(self.config.x_range)
        for i in range(d_in):
            lo, hi = x_range[i] if i < len(x_range) else x_range[-1]
            lo_f = float(lo)
            hi_f = float(hi)
            centers.append(0.5 * (lo_f + hi_f))
            scales.append(max(0.5 * (hi_f - lo_f), 1e-12))
        return jnp.asarray(centers), jnp.asarray(scales)

    def _normalize_state_inputs(self, src):
        d_in = int(self.config.d_in)
        center = jnp.asarray(self._state_norm_center, dtype=src.dtype)
        scale = jnp.asarray(self._state_norm_scale, dtype=src.dtype)
        state_norm = (src[..., :d_in] - center) / scale
        if src.shape[-1] == d_in:
            return state_norm
        return jnp.concatenate((state_norm, src[..., d_in:]), axis=-1)

    def setup(self):
        activation_is_module = False
        use_siren = False
        self._state_norm_center, self._state_norm_scale = self._build_state_normalization_stats()

        match self.config.activation:
            case "wave":
                self.activation = WaveAct
                activation_is_module = True
            case "tanh":
                self.activation = nn.tanh
            case "swish":
                self.activation = nn.swish
            case "sine":
                self.activation = ScaledSine
                activation_is_module = True
                use_siren = True
            case _:
                raise Exception("Activation '"+self.config.activation+"' Not Implemented")

        layers = []

        if self.config.four_emb:
            self.four_layer = FourierEmbs(self.config)

        for i in range(self.config.num_layers):
            if use_siren:
                kernel_init = self._siren_first_layer_init() if i == 0 else self._siren_hidden_layer_init()
                layers.append(nn.Dense(self.config.d_hidden, kernel_init=kernel_init))
            else:
                layers.append(nn.Dense(self.config.d_hidden))

            if activation_is_module:
                if use_siren:
                    layers.append(self.activation(self.config.sine_omega0))
                else:
                    layers.append(self.activation())
            else:
                layers.append(self.activation)
        
        self.output_layer = nn.Dense(self.config.d_out)

        self.layers = layers

    def _prepare_model_input(self, *args):
        src = args[0]
        for i in args[1:len(args)]:
            src = jnp.concatenate((src, i), axis=-1)

        if bool(getattr(self.config, "input_normalization", False)):
            src = self._normalize_state_inputs(src)

        if self.config.periodic:
            for idx in self.config.periodic_idx:
                idx_int = int(idx)
                theta = jnp.copy(src[..., idx_int])
                if bool(getattr(self.config, "input_normalization", False)) and idx_int < int(self.config.d_in):
                    scale = jnp.asarray(self._state_norm_scale[idx_int], dtype=src.dtype)
                    center = jnp.asarray(self._state_norm_center[idx_int], dtype=src.dtype)
                    theta = theta * scale + center
                src = src.at[..., idx_int].set(jnp.sin(theta))
                src = jnp.concatenate((src, jnp.cos(theta[..., jnp.newaxis])), axis=-1)

        return src
    
    def __call__(self,*args):
        src = self._prepare_model_input(*args)

        if self.config.four_emb:
            src = self.four_layer(src)

        i = 1
        src_skip = self.layers[0](src)

        for layer in self.layers:
            src = layer(src)
            if self.config.skip_conn:
                src = jax.lax.select(jnp.any(i == jnp.asarray(self.config.skip_layers) * 2), src + src_skip, src)
                src_skip = jax.lax.select(jnp.any(i == jnp.asarray(self.config.save_layers) * 2), src, src_skip)
            i += 1
        
        src = self.output_layer(src)
        
        return src
