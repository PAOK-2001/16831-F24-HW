import numpy as np

from rob831.agents.base_agent import BaseAgent
from rob831.policies.MLP_policy import MLPPolicyPG
from rob831.infrastructure.replay_buffer import ReplayBuffer

from rob831.infrastructure.utils import normalize, unnormalize

class PGAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super().__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.nn_baseline = self.agent_params['nn_baseline']
        self.reward_to_go = self.agent_params['reward_to_go']
        self.gae_lambda = self.agent_params['gae_lambda']

        # actor/policy
        self.actor = MLPPolicyPG(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=self.agent_params['nn_baseline']
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(1000000)

    def train(self, observations, actions, rewards_list, next_observations, terminals):

        """
            Training a PG agent refers to updating its actor using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.
        """

        qvals = self.calculate_q_vals(rewards_list)
        advantages = self.estimate_advantage(observations, rewards_list, qvals, terminals)
        train_log = self.actor.update(observations, actions, advantages, qvals)

        return train_log


    def calculate_q_vals(self, rewards_list):

        """
            Monte Carlo estimation of the Q function.
            - Input: rewards_list, where rewards_list[i] is an array of the rewards
            - Output: q_values, where q_values[i] is an array of the estimated Q values
        """
        # Estimate Q^{pi}(s_t, a_t) by the total discounted reward summed over entire trajectory
        if not self.reward_to_go:
            #use the whole traj for each timestep
            q_values = np.concatenate([self._discounted_return(rews) for rews in rewards_list])

        # Case 2: reward-to-go PG
        # Estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting from t
        else:
            q_values = np.concatenate([self._discounted_cumsum(rews) for rews in rewards_list])

        # Q_values should be a 1D numpy array where the indices correspond to the same ordering as observations, actions, etc.
        return q_values

    def estimate_advantage(self, obs, rewards_list, q_values, terminals):

        """
            Computes advantages by (possibly) using GAE, or subtracting a baseline from the estimated Q values
        """

        # Estimate the advantage when nn_baseline is True,
        # by querying the neural network that you're using to learn the value function
        if self.nn_baseline:
            values_normalized = self.actor.run_baseline_prediction(obs)
            assert values_normalized.ndim == q_values.ndim
            # Values were trained with standardized q_values, so ensure that the predictions have 
            # the same mean and standard deviation as the current batch of q_values. ie renormalize
            values = unnormalize(values_normalized, np.mean(values_normalized), np.std(values_normalized))
            values = normalize(values, np.mean(q_values), np.std(q_values))
            
            if self.gae_lambda is not None:
                ## append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])
                ## combine rews_list into a single array
                rewards = np.concatenate(rewards_list)

                ## create empty numpy array to populate with GAE advantage estimates, with dummy T+1 value for simpler recursive calculation
                batch_size = obs.shape[0]
                advantages = np.zeros(batch_size + 1)

                prev_gae = 0
                # Recursively compute advantage estimates starting from timestep T.
                for i in reversed(range(batch_size)):
                    # # Use terminals to handle edge cases. terminals[i]
                    # if terminals[i]: # is 1 if the state is the last in its trajectory, and
                    #     delta = rewards[i] - self.gamma * values[i + 1]
                    #     prev_gae = delta

                    # else: ## 0 otherwise.
                    #     delta = rewards[i] + self.gamma * values[i + 1] - values[i]
                    #     prev_gae = delta + self.gamma * self.gae_lambda * prev_gae
                    
                    # NOTE: alternatively, could use the following code to handle edge cases
                    delta = rewards[i] + self.gamma * values[i + 1]   * (1 - terminals[i]) - values[i]
                    prev_gae = delta   + self.gamma * self.gae_lambda * prev_gae * (1 - terminals[i])
                    advantages[i] = prev_gae

                # # remove dummy advantage
                advantages = advantages[:-1]

            else:
                advantages = q_values - values

        # Else, just set the advantage to [Q]
        else:
            advantages = q_values.copy()

        # Normalize the resulting advantages
        if self.standardize_advantages:
            # Standardize the advantages to have a mean of zero and a standard deviation of one
            advantages = normalize(advantages, np.mean(advantages), np.std(advantages))

        return advantages

    #####################################################
    #####################################################

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)

    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################

    def _discounted_return(self, rewards):
        """
            Helper function

            Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T

            Output: array where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
        """
        rewards = np.array(rewards)
        discounted_returns = np.zeros(len(rewards))

        for t in range(len(rewards)):
            for t_prime in range(0, len(rewards)):
                discounted_coeff = self.gamma ** t_prime
                discounted_returns[t] += discounted_coeff * rewards[t_prime]
        
        discounted_returns = np.array(discounted_returns)
        return discounted_returns

    def _discounted_cumsum(self, rewards):
        """
            Helper function which
            -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
            -and returns an array where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """
        rewards = np.array(rewards)
        discounted_cumsums = np.zeros(len(rewards))

        for t in range(len(rewards)):
            for t_prime in range(t, len(rewards)):
                discounted_coeff = self.gamma ** (t_prime - t)
                discounted_cumsums[t] += discounted_coeff * rewards[t_prime]

        return discounted_cumsums
