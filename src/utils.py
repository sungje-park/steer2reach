import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as onp

@jax.tree_util.register_pytree_node_class
class Key:
    def __init__(self,key):
        self.key = key

    @classmethod
    def create_key(cls,seed):
        temp = cls.__new__(cls)
        temp.__init__(jax.random.key(seed))
        return temp
    
    # JAX PyTree Definitions
    def tree_flatten(self):
        children = (self.key,)
        aux_data = {}
        return (children,aux_data)

    @classmethod
    def tree_unflatten(cls,aux_data,children):
        return cls(*children, **aux_data)
    
    def newkey(self):
        self.key,ret_key = jax.random.split(self.key)
        return ret_key
    
    def change(self):
        self.key = self.newkey()

    def split(self,num):
        keys = jax.random.split(self.key,num)
        return [Key(k) for k in keys]