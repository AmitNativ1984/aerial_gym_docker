from rl_games.algos_torch.network_builder import NetworkBuilder
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import torch
from typing import Tuple


class SNNNetworkBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self, params):
        """Called when config is loaded - extract SNN params from YAML """
        self.snn_config = params.get("snn",{})

    def build(self, name, **kwargs):
        """Build and return the actual network """

        return SNNActorCriticNetwork(
            input_dim=kwargs["input_shape"][0],
            action_dim=kwargs["actions_num"],
            **self.snn_config
        )


class SNNActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, **snn_config):
        """
        Spiking Actor-Critic Network Using LIF Neurons

        Parameters:
        - input_dim (int): Dimension of the input observation space.
        - action_dim (int): Dimension of the action space.
        - snn_config (dict): Configuration parameters for the SNN architecture, with the following keys:
            - hidden_dim (int): hidden layers dim. (currently all the same).
            - spike_grad (str): Type of surrogate gradient function to use.
            - learn_beta (bool): Whether to learn the membrane potential decay factor.
            - beta (float): Initial value for the membrane potential decay factor.
            - reset_mechanism (str): Type of reset mechanism after spike.
            - reset_delay (int): Delay steps for reset mechanism.
            - threshold (float): Neuron firing threshold.
            - learn_threshold (bool): Whether to learn the firing threshold.

        """

        super(SNNActorCriticNetwork, self).__init__()

        # Select surrogate gradient function
        if snn_config["spike_grad"] == "sigmoid":
            spike_grad = surrogate.sigmoid(slope=25)
        elif snn_config["spike_grad"] == "atan":
            spike_grad = surrogate.atan(alpha=2.0)
        elif snn_config["spike_grad"] == "fast_sigmoid":
            spike_grad = surrogate.fast_sigmoid(slope=25)
        else:
            raise ValueError(f"Unsupported spike_grad: {snn_config['spike_grad']}")


        self.num_steps = snn_config["num_steps"]

        self.features_extractor = nn.Flatten()

        # Policy Network
        self.policy_fc1 = nn.Linear(in_features=input_dim, out_features=snn_config["hidden_dim"])
        self.policy_lif1 = snn.Leaky(beta=snn_config["beta"],
                                     reset_mechanism=snn_config["reset_mechanism"],
                                     reset_delay=snn_config["reset_delay"],
                                     spike_grad=spike_grad
                                     )
        self.policy_fc2 = nn.Linear(in_features=snn_config["hidden_dim"], out_features=snn_config["hidden_dim"])
        self.policy_lif2 = snn.Leaky(beta=snn_config["beta"],
                                     reset_mechanism=snn_config["reset_mechanism"],
                                     reset_delay=snn_config["reset_delay"],
                                     spike_grad=spike_grad)
        self.policy_fc3 = nn.Linear(in_features=snn_config["hidden_dim"], out_features=snn_config["hidden_dim"])
        self.policy_lif3 = snn.Leaky(beta=snn_config["beta"],
                                     reset_mechanism=snn_config["reset_mechanism"],
                                     reset_delay=snn_config["reset_delay"],
                                     spike_grad=spike_grad)

        # Action head: converts the snn latent spikes (hidden_dim) to action mean (action_dim)
        self.action_head = nn.Linear(in_features=snn_config["hidden_dim"], out_features=action_dim)

        # Sigma: learnable parameter, one per action
        # Initialized to 0, so initial std = exp(0) = 1.0
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Value Network (also SNN)
        self.value_fc1 = nn.Linear(in_features=input_dim, out_features=snn_config["hidden_dim"])
        self.value_lif1 = snn.Leaky(beta=snn_config["beta"],
                                    reset_mechanism=snn_config["reset_mechanism"],
                                    reset_delay=snn_config["reset_delay"],
                                    spike_grad=spike_grad)
        self.value_fc2 = nn.Linear(in_features=snn_config["hidden_dim"], out_features=snn_config["hidden_dim"])
        self.value_lif2 = snn.Leaky(beta=snn_config["beta"],
                                    reset_mechanism=snn_config["reset_mechanism"],
                                    reset_delay=snn_config["reset_delay"],
                                    spike_grad=spike_grad)
        self.value_fc3 = nn.Linear(in_features=snn_config["hidden_dim"], out_features=snn_config["hidden_dim"])
        self.value_lif3 = snn.Leaky(beta=snn_config["beta"],
                                    reset_mechanism=snn_config["reset_mechanism"],
                                    reset_delay=snn_config["reset_delay"],
                                    spike_grad=spike_grad)

        # Value head: converts latent spikes (hidden_dim) to value estimate (1)
        self.value_head = nn.Linear(in_features=snn_config["hidden_dim"], out_features=1)

    def is_rnn(self):
        """Required by rl_games - indicates this is not an RNN network."""
        return False

    def forward(self, obs_dict) -> Tuple[torch.tensor, torch.tensor, torch.tensor, None]:
        """
        Forward pass over multiple time steps.

        Parameters:
            - obs_dict (dict) containing the observations

        Returns:
           - mu (tensor)
           - sigma (tensor)
           - value (tensor)
           - states (tensor) = None
        """

        x = obs_dict["obs"]

        # Initialize membrane potentials for policy SNN:
        policy_mem1 = self.policy_lif1.reset_mem()
        policy_mem2 = self.policy_lif2.reset_mem()
        policy_mem3 = self.policy_lif3.reset_mem()

        # Initialize membrane potentials for value SNN:
        value_mem1 = self.value_lif1.reset_mem()
        value_mem2 = self.value_lif2.reset_mem()
        value_mem3 = self.value_lif3.reset_mem()

        spikes_policy_acc = []
        spikes_value_acc = []

        x = self.features_extractor(x)

        # === Iterate over timesteps ===
        for t in range(self.num_steps):
            # Policy network - Layer 1
            policy_fc1_out = self.policy_fc1(x)
            policy_spk1, policy_mem1 = self.policy_lif1(policy_fc1_out, policy_mem1)

            # Policy network - Layer 2
            policy_fc2_out = self.policy_fc2(policy_spk1)
            policy_spk2, policy_mem2 = self.policy_lif2(policy_fc2_out, policy_mem2)

            # Policy network - Layer 3
            policy_fc3_out = self.policy_fc3(policy_spk2)
            policy_spk3, policy_mem3 = self.policy_lif3(policy_fc3_out, policy_mem3)

            spikes_policy_acc.append(policy_spk3)

            # Value network - Layer 1
            value_fc1_out = self.value_fc1(x)
            value_spk1, value_mem1 = self.value_lif1(value_fc1_out, value_mem1)

            # Value network - Layer 2
            value_fc2_out = self.value_fc2(value_spk1)
            value_spk2, value_mem2 = self.value_lif2(value_fc2_out, value_mem2)

            # Value network - Layer 3
            value_fc3_out = self.value_fc3(value_spk2)
            value_spk3, value_mem3 = self.value_lif3(value_fc3_out, value_mem3)

            spikes_value_acc.append(value_spk3)

        policy_mean_spikes = torch.stack(spikes_policy_acc).mean(dim=0)
        value_mean_spikes = torch.stack(spikes_value_acc).mean(dim=0)

        mu = self.action_head(policy_mean_spikes)   # (batch, action_dim)
        log_std = mu * 0 + self.log_std             # (batch, action_dim)
        value = self.value_head(value_mean_spikes)  # (batch, 1)
        states = None

        return mu, log_std, value, states
