from tsp import __version__
import numpy as np
from tsp import thompson_sample, MABandit
import pytest


def test_version():
    assert __version__ == '0.1.0'


def test_MABandit_object():
    success_rates = [0.1, 0.2, 0.3]
    bandit = MABandit(success_rates)

    # attributes are set correctly
    assert bandit.success_rates == success_rates
    assert bandit.number_of_arms == len(success_rates)
    # playing a valid arm returns either 0 or 1
    assert all(bandit.play(i) in [0, 1] for i in range(len(success_rates)))
    # playing an invalid arm raises an AssertionError
    with pytest.raises(AssertionError):
        bandit.play(3)


def test_thompson_sampling():
    success_rates = [0.1, 0.2, 0.3]
    bandit = MABandit(success_rates)
    n_trials = 10
    ab, played_arms = thompson_sample(bandit, n_trials)

    # shape of ab and played_arms is as expected
    assert ab.shape == (n_trials, len(success_rates), 2)
    assert played_arms.shape == (n_trials, )
    # a or b is increased by one in each update, so the sum of all values of a and b of all slot machines/arms
    # must equal to the number of trials. We have to subtract one because the final update is not written to ab and we
    # have to subtract twice the amount of machines/arms because a and b are initialised with 1 for all machines:
    assert sum(ab[-1, :, 0] + ab[-1, :, 1]) == n_trials - 1 + 2*len(success_rates)


def test_random_seed_thompson_sampling():
    success_rates = [0.1, 0.2, 0.3]
    bandit = MABandit(success_rates)
    n_trials = 10
    # sampling twice with the same seed gives the same results
    ab_first, _ = thompson_sample(bandit, n_trials, seed=1)
    ab_second, _ = thompson_sample(bandit, n_trials, seed=1)

    np.testing.assert_array_equal(ab_first, ab_second)
