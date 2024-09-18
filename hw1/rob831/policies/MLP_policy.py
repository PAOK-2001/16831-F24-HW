import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch

from rob831.policies.base_policy import BasePolicy
from rob831.infrastructure import pytorch_util as ptu

class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers, size=self.size,
            )
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # TODO return the action that the policy prescribes
        observation = ptu.from_numpy(observation)
        ac_prob = self.forward(observation)
        action  = ac_prob.sample()
        action  = action.cpu().numpy()
        return action

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        # Compare predicted actions to labelled actions
        raise NotImplementedError

    def forward(self, observation: torch.FloatTensor) -> Any:
        # For discrete polcy we have a discrete distribution given by the logits
        if self.discrete:
            logits = self.logits_na(observation)
            ac_prob = torch.distributions.Categorical(logits=logits)

        # For continous actions the have a normal distribution, so output of network is mean and std.
        else:
            ac_mean = self.mean_net(observation)
            ac_std  = torch.exp(self.logstd)
            ac_prob = torch.distributions.Normal(ac_mean, ac_std)
        return ac_prob

class MLPPolicySL(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.loss = nn.MSELoss()

    def update(
            self, observations, actions,
            adv_n = None, acs_labels_na=None, qvals=None
    ):

        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)

        pred_actions = self.forward(observations)

        # TODO: update the policy and return the loss
        if self.discrete:
            pred_actions = pred_actions.logits
            actions = actions.long()

        else:
            pred_actions = pred_actions.rsample()

        self.optimizer.zero_grad()
        
        loss = self.loss(actions, pred_actions)
        loss.backward()

        self.optimizer.step()

        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }
