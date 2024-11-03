import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        # Return the action that maxinmizes the Q-value at the current observation as the output
        # action = torch.argmax(self.critic.q_net(observation), dim=1, keepdim=True)
        action_val = self.critic.q_net(observation).cpu().detach().numpy()
        action = np.argmax(action_val, axis=1, keepdims=True)
        return action.squeeze()
