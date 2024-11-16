import numpy as np

from .base_policy import BasePolicy


class MPCPolicy(BasePolicy):

    def __init__(self,
                 env,
                 ac_dim,
                 dyn_models,
                 horizon,
                 N,
                 sample_strategy='random',
                 cem_iterations=4,
                 cem_num_elites=5,
                 cem_alpha=1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None  # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

        # Sampling strategy
        allowed_sampling = ('random', 'cem')
        assert sample_strategy in allowed_sampling, f"sample_strategy must be one of the following: {allowed_sampling}"
        self.sample_strategy = sample_strategy
        self.cem_iterations = cem_iterations
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha

        print(f"Using action sampling strategy: {self.sample_strategy}")
        if self.sample_strategy == 'cem':
            print(f"CEM params: alpha={self.cem_alpha}, "
                + f"num_elites={self.cem_num_elites}, iterations={self.cem_iterations}")

    def sample_action_sequences(self, num_sequences, horizon, obs=None):

        if self.sample_strategy == 'random' \
            or (self.sample_strategy == 'cem' and obs is None):

            # Uniformly sample trajectories
            random_action_sequences = np.random.uniform(
                low=self.low,
                high=self.high,
                size=(num_sequences, horizon, self.ac_dim)
            ) # (N, H, D_action)
            return random_action_sequences
        
        elif self.sample_strategy == 'cem':
            # TODO(Q5): Implement action selection using CEM.
            # Begin with randomly selected actions, then refine the sampling distribution
            # iteratively as described in Section 3.3, "Iterative Random-Shooting with Refinement" of
            # https://arxiv.org/pdf/1909.11652.pdf 
            mean = np.zeros((horizon, self.ac_dim))
            variance =  np.ones((horizon, self.ac_dim))

            for iter in range(self.cem_iterations):
       
                cem_action = np.random.normal(
                            loc=mean, 
                            scale=np.sqrt(variance), 
                            size=(num_sequences, horizon, self.ac_dim))
                cem_action = np.clip(cem_action, self.low, self.high)
                    
                # Get current elite mean and variance
                action_scores = self.evaluate_candidate_sequences(cem_action, obs)
                idx = np.argsort(action_scores)[-self.cem_num_elites:]
                elite_actions = cem_action[idx] 

                mean     = self.cem_alpha * np.mean(elite_actions, axis=0) + (1 - self.cem_alpha) * mean
                variance = self.cem_alpha * np.var(elite_actions, axis=0) + (1 - self.cem_alpha) * mean
                

            cem_action = mean
            return cem_action[None]
        else:
            raise Exception(f"Invalid sample_strategy: {self.sample_strategy}")

    def evaluate_candidate_sequences(self, candidate_action_sequences, obs):
        sum_of_rewards = []
        for model in self.dyn_models:
            sum_of_rewards.append(self.calculate_sum_of_rewards(obs, candidate_action_sequences, model))        
        # Return the mean predictions across all ensembles
        sum_of_rewards = np.array(sum_of_rewards)
        mean_rewards = np.mean(sum_of_rewards, axis=0)
        return mean_rewards


    def get_action(self, obs):
        if self.data_statistics is None:
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        # sample random actions (N x horizon x action_dim)
        candidate_action_sequences = self.sample_action_sequences(
            num_sequences=self.N, horizon=self.horizon, obs=obs)

        if candidate_action_sequences.shape[0] == 1:
            # CEM: only a single action sequence to consider; return the first action
            return candidate_action_sequences[0][0][None]
        else:
            predicted_rewards = self.evaluate_candidate_sequences(candidate_action_sequences, obs)
            max_idx = np.argmax(predicted_rewards) # TODO: check if this is argmin or argmax. I beliebe it should maximse the rewards
            best_action_sequence = candidate_action_sequences[max_idx]
            action_to_take = best_action_sequence[0]
            return action_to_take[None]  # Unsqueeze the first index

    def calculate_sum_of_rewards(self, obs, candidate_action_sequences, model):
        """
        :param obs: numpy array with the current observation. Shape [D_obs]
        :param candidate_action_sequences: numpy array with the candidate action
        sequences. Shape [N, H, D_action] where
            - N is the number of action sequences considered
            - H is the horizon
            - D_action is the action of the dimension
        :param model: The current dynamics model.
        :return: numpy array with the sum of rewards for each action sequence.
        The array should have shape [N].
        """
        N, H, _ = candidate_action_sequences.shape
        sum_of_rewards = np.zeros(N)
        curr_obs = np.tile(obs, (N, 1))  # Shape (N, D_obs)  

        for t in range(H):
            curr_action = candidate_action_sequences[:, t, :]
            next_obs = model.get_prediction(curr_obs, curr_action, self.data_statistics)
            reward, _ = self.env.get_reward(curr_obs, curr_action)
            # Update the sum of rewards
            sum_of_rewards += reward
            curr_obs = next_obs

        return sum_of_rewards