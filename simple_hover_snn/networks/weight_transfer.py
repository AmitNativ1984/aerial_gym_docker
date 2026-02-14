"""
Weight Transfer Functions for MLP to SNN conversion.

Transfers trained MLP weights to SNN network for fine-tuning.
The linear layers have identical shapes but different parameter names.
"""

import torch
from typing import Dict


# Mapping from MLP parameter names to SNN parameter names
MLP_TO_SNN_MAPPING = {
    # Policy network layers (MLP uses Sequential, SNN uses named layers)
    "policy_net.0.weight": "policy_fc1.weight",
    "policy_net.0.bias": "policy_fc1.bias",
    "policy_net.2.weight": "policy_fc2.weight",
    "policy_net.2.bias": "policy_fc2.bias",
    "policy_net.4.weight": "policy_fc3.weight",
    "policy_net.4.bias": "policy_fc3.bias",

    # Value network layers
    "value_net.0.weight": "value_fc1.weight",
    "value_net.0.bias": "value_fc1.bias",
    "value_net.2.weight": "value_fc2.weight",
    "value_net.2.bias": "value_fc2.bias",
    "value_net.4.weight": "value_fc3.weight",
    "value_net.4.bias": "value_fc3.bias",

    # Output heads (same names - direct copy)
    "action_head.weight": "action_head.weight",
    "action_head.bias": "action_head.bias",
    "value_head.weight": "value_head.weight",
    "value_head.bias": "value_head.bias",

    # Log std (same name - direct copy)
    "log_std": "log_std",
}


def transfer_mlp_to_snn_weights(mlp_state_dict: Dict[str, torch.Tensor],
                                 snn_state_dict: Dict[str, torch.Tensor],
                                 verbose: bool = True) -> Dict[str, torch.Tensor]:
    """
    Transfer weights from trained MLP to SNN network.

    Args:
        mlp_state_dict: State dict from trained MLP model
        snn_state_dict: State dict from SNN model (will be modified)
        verbose: Print transfer details

    Returns:
        Updated SNN state dict with MLP weights transferred
    """
    transferred = []
    skipped_mlp = []
    skipped_snn = []

    # Transfer weights using mapping
    for mlp_name, snn_name in MLP_TO_SNN_MAPPING.items():
        if mlp_name in mlp_state_dict and snn_name in snn_state_dict:
            mlp_param = mlp_state_dict[mlp_name]
            snn_param = snn_state_dict[snn_name]

            # Verify shapes match
            if mlp_param.shape != snn_param.shape:
                raise ValueError(
                    f"Shape mismatch for {mlp_name} -> {snn_name}: "
                    f"MLP {mlp_param.shape} vs SNN {snn_param.shape}"
                )

            snn_state_dict[snn_name] = mlp_param.clone()
            transferred.append(f"{mlp_name} -> {snn_name}")
        elif mlp_name not in mlp_state_dict:
            skipped_mlp.append(mlp_name)
        elif snn_name not in snn_state_dict:
            skipped_snn.append(snn_name)

    if verbose:
        print(f"\n=== MLP to SNN Weight Transfer ===")
        print(f"Transferred {len(transferred)} parameters:")
        for t in transferred:
            print(f"  ✓ {t}")

        if skipped_mlp:
            print(f"\nMissing from MLP checkpoint ({len(skipped_mlp)}):")
            for s in skipped_mlp:
                print(f"  ✗ {s}")

        if skipped_snn:
            print(f"\nMissing from SNN model ({len(skipped_snn)}):")
            for s in skipped_snn:
                print(f"  ✗ {s}")

        # List SNN-only parameters (LIF neurons, etc.)
        snn_only = [k for k in snn_state_dict.keys()
                    if k not in MLP_TO_SNN_MAPPING.values()]
        if snn_only:
            print(f"\nSNN-only parameters (not transferred, using defaults):")
            for s in snn_only:
                print(f"  • {s}")

        print(f"\n=== Transfer Complete ===\n")

    return snn_state_dict


def load_mlp_checkpoint_to_snn(mlp_checkpoint_path: str,
                                snn_model: torch.nn.Module,
                                device: str = "cuda",
                                verbose: bool = True) -> torch.nn.Module:
    """
    Load MLP checkpoint and transfer weights to SNN model.

    Args:
        mlp_checkpoint_path: Path to MLP checkpoint (.pth file)
        snn_model: SNN model instance to load weights into
        device: Device to load checkpoint on
        verbose: Print transfer details

    Returns:
        SNN model with transferred weights
    """
    # Load MLP checkpoint
    checkpoint = torch.load(mlp_checkpoint_path, map_location=device)

    # RL Games checkpoints store model in 'model' key
    if 'model' in checkpoint:
        mlp_state_dict = checkpoint['model']
    else:
        mlp_state_dict = checkpoint

    # Get current SNN state dict
    snn_state_dict = snn_model.state_dict()

    # Transfer weights
    updated_state_dict = transfer_mlp_to_snn_weights(
        mlp_state_dict, snn_state_dict, verbose=verbose
    )

    # Load updated state dict into SNN model
    snn_model.load_state_dict(updated_state_dict)

    return snn_model


def verify_weight_transfer(mlp_state_dict: Dict[str, torch.Tensor],
                           snn_state_dict: Dict[str, torch.Tensor]) -> bool:
    """
    Verify that weights were correctly transferred from MLP to SNN.

    Args:
        mlp_state_dict: Original MLP state dict
        snn_state_dict: SNN state dict after transfer

    Returns:
        True if all mapped weights match exactly
    """
    all_match = True

    for mlp_name, snn_name in MLP_TO_SNN_MAPPING.items():
        if mlp_name in mlp_state_dict and snn_name in snn_state_dict:
            mlp_param = mlp_state_dict[mlp_name]
            snn_param = snn_state_dict[snn_name]

            if not torch.allclose(mlp_param, snn_param):
                print(f"Mismatch: {mlp_name} -> {snn_name}")
                all_match = False

    if all_match:
        print("✓ All weights transferred correctly")

    return all_match
