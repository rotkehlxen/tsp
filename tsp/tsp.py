import numpy as np
from scipy.stats import bernoulli
from typing import Tuple, Optional


class MABandit:
    """
    This class simulates a multi-armed bandit. A bandit object is initialised with a list of success probabilities,
    where the length of this list corresponds to the "number of arms" of the bandit.
    """

    def __init__(self, success_rates):
        self.success_rates = success_rates

    @property
    def number_of_arms(self) -> int:
        """ The number of arms of the multi-armed bandit. (Or the number of different slot machines) """
        return len(self.success_rates)

    def play(self, arm: int) -> int:
        """
        Play the selected [arm] of MABandit. Returns 1 with success probability success_rates[arm], or zero otherwise
        (Bernoulli distribution).

        :param arm:  Int, ID of the arm, takes values between 0 and len(success_rates) - 1
        :return:     Int, either 0 or 1 according to Bernoulli distribution
        """
        assert 0 <= arm < self.number_of_arms, f"Only arm IDs from 0 .. {self.number_of_arms - 1} can be selected!"
        return bernoulli.rvs(self.success_rates[arm])


def thompson_sample(bandit: MABandit, n_trials: int, seed: Optional[int] = None) -> Tuple:
    """
    Thompson sampling for solving a multi-armed bandit problem. In short, we assume that the unknown success probabilities
    of the arms of the bandit are distributed according to a Beta distribution. In each iteration, we sample success
    probabilities from these distributions and play the arm with the highest estimated chance of winning. We register
    our wins and losses by updating the prior beta distributions to posterior beta distributions. As the posterior
    distributions become more sharply peaked, we transit from an exploration phase to exploitation of bandits with
    higher chance of winning.

    :param bandit:    class MABandit, a multi-armed bandit
    :param n_trials:  Int, the number of rounds that we can play the multi-armed bandit (our limited resource)
    :param seed:      Int, random seed
    :return:          Tuple, numpy array "ab" with parameters a and b of the beta distribution for each arm and every
                             single trial with shape (n_trials, number of arms, 2) and values of a in ab[:, :, 0] and
                             values of b in ab[:, :, 1] AND "arms", numpy array with index of the arm that was played
                             in each trial with shape (n_trials, )
    """

    if seed:
        np.random.seed(seed)

    n_arms = bandit.number_of_arms
    # preallocate return values of this function, init hyper-parameters of all beta distributions with 1
    # (uniform distribution)
    ab = np.ones((n_trials + 1, n_arms, 2))
    arms = np.zeros(n_trials)

    for i in range(n_trials):
        # sample success rates from beta distribution
        estimated_success_rates = np.random.beta(ab[i, :, 0], ab[i, :, 1])
        # pick arm/slot machine with highest success rate
        arm = np.argmax(estimated_success_rates)
        # play the selected arm/machine and observe outcome
        outcome = bandit.play(arm)
        # save which arm was played in each round
        arms[i] = arm
        # update hyper-parameters
        ab[i + 1, :, :] = ab[i, :, :]
        ab[i + 1, arm, 0] = ab[i, arm, 0] + outcome
        ab[i + 1, arm, 1] = ab[i, arm, 1] + (1 - outcome)

    return ab[:-1, :, :], arms
