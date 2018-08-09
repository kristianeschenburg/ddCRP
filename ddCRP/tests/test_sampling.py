import numpy as np
from ddCRP import sampling


def test_gibbs():
    """
    Method for testing Gibbs sampling.
    """

    lp = np.asarray([0, 1])

    expected = 1
    actual = sampling.Gibbs().sample(lp)
    assert expected == actual


def test_gibbs_null():

    """
    Method for testing null log-probabilities.
    """

    lp = np.zeros((3))
    expected = 0
    actual = sampling.Gibs().sample(lp)

    assert expected == actual
