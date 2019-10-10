from abc import ABC, abstractmethod


class Prior(ABC):

    """
    Abstract class for prior on feature data in ddCRP clustering model.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def sufficient_statistics(self):

        """
        Method for computing the sufficient statistics of a the model.
        """
        pass

    @abstractmethod
    def posterior_parameters(self):

        """
        Method for computing posterior parameters of the model.
        """
        pass

    @abstractmethod
    def marginal_evidence(self):

        """
        Method for computing marginal evidence of the data,
        given posterior parameters of the model.
        """
        pass

    @abstractmethod
    def full_evidence(self):

        """
        Compute the full marginal evidence of a given clustering.
        """
        pass

