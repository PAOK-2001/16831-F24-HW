import numpy as np
from collections import OrderedDict

from rob831.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from rob831.infrastructure.replay_buffer import ReplayBuffer
from rob831.infrastructure.utils import normalize
import rob831.infrastructure.pytorch_util as  ptu
from rob831.policies.MLP_policy import MLPPolicyAC
from .base_agent import BaseAgent


class ACAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(ACAgent, self).__init__()

        self.env = env
        self.agent_params = agent_params

        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']

        self.actor = MLPPolicyAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
        )
        self.critic = BootstrappedContinuousCritic(self.agent_params)

        self.replay_buffer = ReplayBuffer()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
            loss_critic = self.critic.update(
                ob_no= ob_no, 
                ac_na= ac_na, 
                next_ob_no= next_ob_no,
                reward_n= re_n, 
                terminal_n= terminal_n
            )

        advantage = self.estimate_advantage(
            ob_no= ob_no, 
            next_ob_no= next_ob_no,
            re_n= re_n,
            terminal_n= terminal_n
        )

        for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
            loss_actor = self.actor.update(
                observations = ob_no, 
                actions = ac_na,
                adv_n = advantage
            )


        loss = OrderedDict()
        loss['Loss_Critic'] = loss_critic
        loss['Loss_Actor'] = loss_actor

        return loss

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        # TODO Implement the following pseudocode:
        # 1) query the critic with ob_no, to get V(s)
        # 2) query the critic with next_ob_no, to get V(s')
        # 3) estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
        # HINT: Remember to cut off the V(s') term (ie set it to 0) at terminal states (ie terminal_n=1)
        # 4) calculate advantage (adv_n) as A(s, a) = Q(s, a) - V(s)
        obs = ptu.from_numpy(ob_no)
        next_obs = ptu.from_numpy(next_ob_no)

        v_value = self.critic.forward(obs)
        v_prime = self.critic.forward(next_obs)

        v_value = ptu.to_numpy(v_value)
        v_prime = ptu.to_numpy(v_prime)

        q_value = re_n + self.gamma * v_prime * (1 - terminal_n)

        adv_n = q_value - v_value

        if self.standardize_advantages:
            adv_n = normalize(adv_n, np.mean(adv_n), np.std(adv_n))
        return adv_n

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)
