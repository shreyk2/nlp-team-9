# forward hooks that project out directions/subspaces at a given layer
import torch


# hook that subtracts the DoM direction component from hidden states
class DirectionalAblationHook:

    def __init__(self, direction, layer_index):
        self.direction = torch.tensor(direction, dtype=torch.float32)
        self.layer_index = layer_index
        self.handle = None

    # subtract projection onto the direction: h' = h - (h . r_hat) * r_hat
    def hook_function(self, _module, _inputs, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        direction_on_device = self.direction.to(hidden_states.device)
        hidden_float = hidden_states.float()
        if hidden_float.dim() == 2:
            projection = hidden_float @ direction_on_device
            ablated = hidden_float - projection.unsqueeze(-1) * direction_on_device.unsqueeze(0)
        else:
            projection = torch.einsum("bsd,d->bs", hidden_float, direction_on_device)
            broadcast_direction = direction_on_device.unsqueeze(0).unsqueeze(0)
            ablated = hidden_float - projection.unsqueeze(-1) * broadcast_direction
        ablated = ablated.to(hidden_states.dtype)
        if isinstance(output, tuple):
            return (ablated,) + output[1:]
        return ablated

    def attach(self, model):
        target_layer = model.model.layers[self.layer_index]
        self.handle = target_layer.register_forward_hook(self.hook_function)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# hook that subtracts the ActSVD subspace component from hidden states
class SubspaceAblationHook:

    def __init__(self, subspace_basis, layer_index):
        self.subspace_basis = torch.tensor(subspace_basis, dtype=torch.float32)
        self.layer_index = layer_index
        self.handle = None

    def hook_function(self, _module, _inputs, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        basis_on_device = self.subspace_basis.to(hidden_states.device)
        hidden_float = hidden_states.float()
        coefficients = hidden_float @ basis_on_device.T
        projection_vectors = coefficients @ basis_on_device
        ablated = hidden_float - projection_vectors
        ablated = ablated.to(hidden_states.dtype)
        if isinstance(output, tuple):
            return (ablated,) + output[1:]
        return ablated

    def attach(self, model):
        target_layer = model.model.layers[self.layer_index]
        self.handle = target_layer.register_forward_hook(self.hook_function)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class NoOpHook:

    def __init__(self):
        self.handle = None

    def attach(self, model):
        pass

    def remove(self):
        pass
