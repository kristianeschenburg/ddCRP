import numpy as np
from ddCRP import statistics

from sklearn.metrics import normalized_mutual_info_score as NMI


def test_normalize():
    """
    Test to check feature normalization.
    """

    data = np.asarray(
            [[-1, 0, 1],
                [1, 2, 3]])

    mu = data.mean(0)
    stdev = data.std(0)

    expected = (data-mu[None, :]) / stdev[None, :]
    actual = statistics.Normalize(data)

    assert expected == actual


def test_NMI():

    label_true = np.asarray([0, 1, 2, 3])
    label_pred = np.asarray([0, 1, 2, 3])

    expected = NMI(label_true, label_pred)
    actual = statistics.NMI(label_true, label_pred)

    assert expected == actual

    label_true = np.asarray([0, 0, 0, 3])
    label_pred = np.asarray([0, 2, 2, 2])

    expected = NMI(label_true, label_pred)
    actual = statistics.NMI(label_true, label_pred)

    assert expected == actual
