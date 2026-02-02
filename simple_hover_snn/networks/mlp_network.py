"""
Standard MLP Actor-Critic Network for comparison with SNN.

This network exactly matches the reference position_setpoint_task MLP architecture:
- MLP layers: [256, 128, 64]
- Activation: ELU
- No mu activation (unbounded output)
- Learned log_std (initialized to 0, so initial std = 1.0)
"""

from rl_games.algos_torch.network_builder import NetworkBuilder
import torch.nn as nn
import torch
from typing import Tuple


class MLPNetworkBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self, params):
        """Called when config is loaded - extract MLP params from YAML"""
        self.mlp_config = params.get("mlp", {})

    def build(self, name, **kwargs):
        """Build and return the actual network"""
        return MLPActorCriticNetwork(
            input_dim=kwargs["input_shape"][0],
            action_dim=kwargs["actions_num"],
            **self.mlp_config
        )


class MLPActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, **mlp_config):
        """
        Standard MLP Actor-Critic Network (matching reference implementation)

        Parameters:
        - input_dim (int): Dimension of the input observation space.
        - action_dim (int): Dimension of the action space.
        - mlp_config (dict): Configuration parameters for the MLP architecture:
            - units (list): Hidden layer sizes (default: [256, 128, 64])
            - activation (str): Activation function (default: "elu")
            - initializer (dict): Weight initialization config
        """
        super(MLPActorCriticNetwork, self).__init__()

        # Extract config parameters
        units = mlp_config.get("units", [256, 128, 64])
        activation_name = mlp_config.get("activation", "elu")

        # Select activation function
        if activation_name == "elu":
            activation = nn.ELU()
        elif activation_name == "relu":
            activation = nn.ReLU()
        elif activation_name == "tanh":
            activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation_name}")

        # Build policy network (actor)
        policy_layers = []
        in_features = input_dim
        for hidden_size in units:
            policy_layers.append(nn.Linear(in_features, hidden_size))
            policy_layers.append(activation)
            in_features = hidden_size
        self.policy_net = nn.Sequential(*policy_layers)

        # Action head: outputs unbounded mu (no activation)
        # Output order: [thrust, roll, pitch, yaw_rate]
        self.action_head = nn.Linear(in_features, action_dim)

        # Learned log_std parameter (initialized to 0, so initial std = exp(0) = 1.0)
        # This matches reference "const_initializer: val: 0"
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Build value network (critic)
        value_layers = []
        in_features = input_dim
        for hidden_size in units:
            value_layers.append(nn.Linear(in_features, hidden_size))
            value_layers.append(activation)
            in_features = hidden_size
        self.value_net = nn.Sequential(*value_layers)

        # Value head: outputs single value estimate
        self.value_head = nn.Linear(in_features, 1)

        # Initialize weights (default PyTorch initialization)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights (matches rl_games default initialization)"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier/Glorot uniform initialization (rl_games default)
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('linear'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def is_rnn(self):
        """Required by rl_games - indicates this is not an RNN network."""
        return False

    def forward(self, obs_dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """
        Forward pass through the network.

        Parameters:
            - obs_dict (dict) containing the observations

        Returns:
           - mu (tensor): Action means (unbounded, no activation)
           - log_std (tensor): Log standard deviations
           - value (tensor): Value estimate
           - states (None): No recurrent state
        """
        x = obs_dict["obs"]

        # Policy network forward pass
        policy_features = self.policy_net(x)
        mu = self.action_head(policy_features)  # Unbounded output (no activation)

        # Broadcast log_std to match batch size
        log_std = self.log_std.unsqueeze(0).expand(mu.shape[0], -1)

        # Value network forward pass
        value_features = self.value_net(x)
        value = self.value_head(value_features)

        states = None

        return mu, log_std, value, states
