import torch.nn as nn

from .layers import LoRALinear


def inject_lora(model: nn.Module, target_names: list[str], r=16, alpha=16):
    """Replaces model's target layers with LoRALinear wrappers. Modifies model in place."""
    to_replace = [] # (parent_name, module_name, module object)

    # create list of layers to be replaced
    for name, module in model.named_modules():
        parent_name, _, module_name = name.rpartition(".") # ("down_blocks.1.1", ".", "q2")
        
        if module_name in target_names and isinstance(module, nn.Linear):
            to_replace.append((parent_name, module_name, module))

    # inject lora layers
    for parent_name, module_name, module in to_replace:
        to_inject = LoRALinear(
            base_layer=module,
            r=r,
            alpha=alpha
        )
        parent = model.get_submodule(parent_name)
        setattr(parent, module_name, to_inject)