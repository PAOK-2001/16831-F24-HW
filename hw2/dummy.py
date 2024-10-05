import numpy as np
gamma = 0.99

def _discounted_return(rewards):
    """
        Helper function

        Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T

        Output: array where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
    """
    rewards = np.array(rewards)
    discounted_returns = np.zeros(len(rewards))

    for t in range(len(rewards)):
        for t_prime in range(0, len(rewards)):
            discounted_coeff = gamma ** t_prime
            discounted_returns[t] += discounted_coeff * rewards[t_prime]
    
    discounted_returns = np.array(discounted_returns)
    return discounted_returns

def _discounted_cumsum(rewards):
    """
        Helper function which
        -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
        -and returns an array where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
    """
    rewards = np.array(rewards)
    discounted_cumsums = np.zeros(len(rewards))

    for t in range(len(rewards)):
        for t_prime in range(t, len(rewards)):
            discounted_coeff = gamma ** (t_prime - t)
            discounted_cumsums[t] += discounted_coeff * rewards[t_prime]

    return discounted_cumsums

if __name__ == "__main__":
    rewards = [i for i in range(10)]
    print(_discounted_return(rewards))
    print(_discounted_cumsum(rewards))
