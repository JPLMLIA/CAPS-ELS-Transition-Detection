from abc import ABCMeta, abstractmethod
from pyhsmm import models as pyhsmm_models
from hmmlearn import hmm as hmmlearn_models
from pyhsmm.util.text import progprint_xrange
from pyhsmm.basic import distributions
import numpy as np


class HMM:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        self.hmm_model = None
        self.data = None
        self.data_dimensions = None
        self.num_datapoints = None
        self.num_states = None

    # Generates samples from the learned distribution for observations for a specific state.
    @abstractmethod
    def generate_samples(self, state, num_samples):
        pass

    # Emission parameters for a given state.
    @abstractmethod
    def emission_params(self, state):
        pass

    # Learn HMM parameters.
    @abstractmethod
    def fit(self, data):
        pass

    # Decode to get most likely sequence of states.
    @abstractmethod
    def predict_state_sequence(self):
        pass

    # Estimate log-likelihood of sequence conditional on learned parameters.
    @abstractmethod
    def log_likelihood(self):
        pass

    # Get posterior distribution over states, conditional on the input.
    @abstractmethod
    def state_distribution(self):
        pass

    # The number of parameters this HMM uses.
    @abstractmethod
    def num_parameters(self):
        pass

    # The Akaike Information Criteria (BIC) function.
    def aic_value(self):
        return 2 * self.num_parameters() - 2 * self.log_likelihood()

    # The Bayesian Information Criteria (BIC) function.
    def bic_value(self):
        return np.log(self.num_datapoints) * self.num_parameters() - 2 * self.log_likelihood()


class VanillaGaussianHMM(HMM):

    def __init__(self, **params):
        HMM.__init__(self)
        self.hmm_model = hmmlearn_models.GaussianHMM(**params)
        self.num_states = params['n_components']

    def generate_samples(self, state, num_samples):
        dist_parameters = self.emission_params(state)
        return distributions.Gaussian(**dist_parameters).rvs(num_samples)

    def emission_params(self, state):
        return {'mu': self.hmm_model.means_[state],
                'sigma': self.hmm_model.covars_[state]}

    def fit(self, data):
        self.data = data
        self.num_datapoints = self.data.shape[0]
        self.data_dimensions = self.data.shape[1]

        self.hmm_model.fit(self.data)

    def predict_state_sequence(self, data=None):
        if data is None:
            data = self.data
        return self.hmm_model.predict(data)

    def log_likelihood(self, data=None):
        if data is None:
            data = self.data
        return self.hmm_model.score(data)

    def state_distribution(self, data=None):
        if data is None:
            data = self.data
        return self.hmm_model.score_samples(data)[1]

    def num_parameters(self):
        # The parameters for the Gaussian HMM are:
        # * a mean and diagonal covariance matrix, for the emission probabilities for each state.
        # * the transition matrix.
        # * the initial distribution over states.
        return self.num_states * (2 * self.data_dimensions) + self.num_states * self.num_states + self.num_states


class VanillaGaussianMixtureHMM(VanillaGaussianHMM):

    def __init__(self, **params):
        HMM.__init__(self)
        self.hmm_model = hmmlearn_models.GMMHMM(**params)
        self.num_states = params['n_components']
        self.num_base_distributions = params['n_mix']

    def num_parameters(self):
         # The parameters for the Gaussian HMM are:
        # * a mean and diagonal covariance matrix, for the emission probabilities for each base distribution in each state.
        # * the transition matrix.
        # * the initial distribution over states.
        return self.num_states * (2 * self.data_dimensions * self.num_base_distributions) + self.num_states * self.num_states + self.num_states


class HDPHMM(HMM):

    def __init__(self, **params):
        HMM.__init__(self)
        self.hmm_model = pyhsmm_models.WeakLimitHDPHMM(**params)
        self.model_samples = None
        self.num_states = len(params['obs_distns'])

    def emission_params(self, state):
        return self.hmm_model.states_list[0].obs_distns[state].params

    def generate_samples(self, state, num_samples):
        return self.hmm_model.states_list[0].obs_distns[state].rvs(num_samples)

    def fit(self, data, num_iterations=1000, verbose=True):
        self.data = data
        self.num_datapoints = self.data.shape[0]
        self.data_dimensions = self.data.shape[1]

        self.hmm_model.add_data(data)

        if verbose:
            index_range = progprint_xrange(num_iterations)
        else:
            index_range = np.arange(num_iterations)

        self.model_samples = [self.hmm_model.resample_and_copy() for index in index_range]

    def predict_state_sequence(self, data=None):
        if data is None:
            data = self.data
        return np.argmax(self.state_distribution(data), axis=1)

    def log_likelihood(self):
        return np.mean([model.log_likelihood() for model in self.model_samples], axis=0)

    def state_distribution(self, data=None):
        if data is None:
            data = self.data
        return np.mean([model.heldout_state_marginals(data) for model in self.model_samples], axis=0)

    def num_parameters(self):
        # The parameters for the HDP-HMM are:
        # * the NIW parameters, for the emission probabilities for each state.
        # * the transition matrix
        # * the initial distribution over states.
        # * the CRP parameters, alpha and gamma.
        return self.num_states * (self.data_dimensions + self.data_dimensions * self.data_dimensions + 2) + self.num_states * self.num_states + self.num_states + 2


class StickyHDPHMM(HDPHMM):

    def __init__(self, **params):
        HMM.__init__(self)
        self.hmm_model = pyhsmm_models.WeakLimitStickyHDPHMM(**params)
        self.num_states = len(params['obs_distns'])

    def num_parameters(self):
        # The parameters for the Sticky HDP-HMM are the same as the HDP-HMM, but with a kappa factor for state persistence.
        return HDPHMM.num_parameters(self) + 1
