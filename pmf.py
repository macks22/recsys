import sys
import logging
import argparse

import theano
import pymc3 as pm
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as stats


def gen_data(n=50, m=20):
    """Generate synthetic data to validate the model with."""
    logging.info('generating (%d, %d) synthetic data matrix' % (n, m))

    # So the model says that we have a number of latent factors, which make up
    # each item, and we have user-specific linear combination weights for these
    # factors. These weights, stored in U, when applied to the latent item
    # factors, stored in V, produce the user's preference for that item. In the
    # PMF model, the priors for U and V are spherical Gaussian priors with a
    # precision matrix equal to the identity vector times the variance, which is
    # a hyperparameter that can be used to control model complexity.
    pass


def rmse(data, predicted):
    """Calculate root mean squared error."""
    I = ~np.isnan(data)   # indicator for missing values
    N = I.sum()           # number of non-missing values
    sqerror = abs(data - predicted) ** 2
    mse = sqerror[I].sum() / N
    return np.sqrt(mse)


class PMF(object):
    """Probabilistic Matrix Factorization"""

    def __init__(self, data, scale, alpha=2, dim=10):
        """
        :param numpy.ndarray data: Observed data to learn the model from.
        :param int alpha: Precision for all observed data.
        :param int dim: Dimensionality, or number of latent factors to learn.
        """
        self.data = data
        self.scale = scale
        self.dim = dim

        # Set precision of U and V according to sample precision.
        self.sample_precision = 1 / np.var(data)
        self.alpha_u = self.alpha_v = self.sample_precision * np.eye(dim)

        # Use fixed precision for the likelihood function.
        self.alpha = np.ones(self.data.shape) * alpha

        # Set None values for attributes to be set later
        self.step = None
        self.trace = None
        self.predicted = None

    @property
    def std(self):
        return np.sqrt(1 / self.alpha[0,0])

    @property
    def model(self):
        try:
            return self._model
        except:
            self.build_model()
            return self._model

    def build_model(self):
        """Construct the model using pymc3 (most of the work is done by theano).
        Note that the `testval` param for U and V initialize the model away from
        0 using a small amount of Gaussian noise.
        """
        n, m = self.data.shape
        dim = self.dim

        logging.info('building the PMF model')
        with pm.Model() as pmf:
            U = pm.MvNormal(
                'U', mu=0, tau=self.alpha_u,
                shape=(n, dim), testval=np.random.randn(n, dim) * .01)
            V = pm.MvNormal(
                'V', mu=0, tau=self.alpha_v,
                shape=(m, dim), testval=np.random.randn(m, dim) * .01)
            R = pm.Normal(
                'R', mu=theano.tensor.dot(U, V.T), tau=self.alpha,
                observed=self.data)

        logging.info('done building PMF model')
        self._model = pmf

    @property
    def map(self):
        try:
            return self._map
        except:
            self.find_map()
            return self._map

    def find_map(self):
        """Use non-gradient based optimization to find MAP."""
        with self.model:
            logging.info('finding MAP using Powell optimization')
            self._map = pm.find_MAP(fmin=sp.optimize.fmin_powell)

    def sample(self, n=100, njobs=4, progressbar=True):
        """Draw n MCMC samples using the NUTS sampler."""
        with self.model:
            logging.info(
                'drawing %d MCMC samples using %d jobs' % (n, njobs))
            self.step = pm.NUTS(scaling=self.map)
            self.trace = pm.sample(n, self.step, start=self.map, njobs=njobs,
                                   progressbar=progressbar)

    def estimate_R(self, U, V):
        R = np.dot(U, V.T)
        n, m = R.shape
        sample_R = np.ndarray(R.shape)
        for i in xrange(n):
            for j in xrange(m):
                sample_R[i,j] = stats.norm.rvs(R[i,j], self.std)

        return sample_R

    def map_rmse(self, test_data):
        sample_R = self.estimate_R(self.map['U'], self.map['V'])
        low, high = self.scale
        sample_R[sample_R < low] = low
        sample_R[sample_R > high] = high
        return rmse(test_data, sample_R)

    def predict(self, burn_in=0):
        self.predicted = np.ndarray(self.data.shape)
        for sample in self.trace[burn_in:]:
            self.predicted += self.estimate_R(sample['U'], sample['V'])

        self.predicted /= len(self.trace[burn_in:])
        low, high = self.scale
        self.predicted[self.predicted < low] = low
        self.predicted[self.predicted > high] = high

    def rmse(self, test_data):
        return rmse(test_data, self.predicted)


class Baseline(object):
    """Calculate baseline predictions."""

    def __init__(self, train_data):
        """Simple heuristic-based transductive learning to fill in missing
        values in data."""
        self.predict(train_data.copy())

    def predict(self, train_data):
        """Fill in missing values with global mean."""
        raise NotImplementedError(
            'baseline prediction not implemented for base class')

    def rmse(self, test_data):
        """Calculate root mean squared error for predictions on test data."""
        return rmse(test_data, self.predicted)


class UniformRandomBaseline(Baseline):
    """Fill missing values with uniform random values."""

    def predict(self, train_data):
        nan_mask = np.isnan(train_data)
        masked_train = np.ma.masked_array(train_data, nan_mask)
        min, max = masked_train.min(), masked_train.max()
        N = nan_mask.sum()
        train_data[nan_mask] = np.random.uniform(min, max, N)
        self.predicted = train_data


class GlobalMeanBaseline(Baseline):
    """Fill in missing values using the global mean."""

    def predict(self, train_data):
        nan_mask = np.isnan(train_data)
        train_data[nan_mask] = train_data[~nan_mask].mean()
        self.predicted = train_data


class MeanOfMeansBaseline(Baseline):
    """Fill in missing values using mean of user/joke/global mean."""

    def predict(self, train_data):
        nan_mask = np.isnan(train_data)
        masked_train = np.ma.masked_array(train_data, nan_mask)
        global_mean = masked_train.mean()
        user_means = masked_train.mean(axis=1)
        joke_means = masked_train.mean(axis=0)
        self.predicted = train_data.copy()
        n, m = train_data.shape
        for i in xrange(n):
            for j in xrange(m):
                if np.ma.isMA(joke_means[j]):
                    self.predicted[i,j] = np.mean(
                        (global_mean, user_means[i]))
                else:
                    self.predicted[i,j] = np.mean(
                        (global_mean, user_means[i], joke_means[j]))


def make_parser():
    parser = argparse.ArgumentParser(
        description='Probabilistic Matrix Factorization.')
    parser.add_argument(
        '-n', '--njobs', type=int, default=4,
        help='number of processes to use for MCMC sampling')
    parser.add_argument(
        '-s', '--samples', type=int, default=100,
        help='number of MCMC samples to draw')
    parser.add_argument(
        '-b', '--burn-in', type=int, default=10,
        help='number of samples to discard as burn-in')
    parser.add_argument(
        '-d', '--dimension', type=int, default=10,
        help='dimensionality of PMF model')
    parser.add_argument(
        '-m', '--method', choices=('ur', 'gm', 'mom', 'pmf-map', 'pmf-mcmc'),
        default='pmf-map',
        help='method to use for making predictions')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='enable verbose logging')
    return parser


def read_jester_data(n=200, m=50):
    """Read a subset of the dense 1000x100 jester dataset.
    :param int n: Number of users to keep.
    :param int m: Number of jokes to keep.
    """
    logging.info('reading data')
    data = pd.read_csv('data/jester-dataset-v1-dense-first-1000.csv')
    data = data.head(n).ix[:,:m]  # get subset for model validation

    logging.info('splitting train/test sets')
    n, m = data.shape           # # users, # jokes
    test_size = m / 10          # use 10% of data as test set
    train_size = m - test_size  # and remainder for training

    train = data.copy()
    train.ix[:,train_size:] = np.nan  # remove test data from train set

    test = data.copy()
    test.ix[:,:train_size] = np.nan  # remove train data from test set

    return train.values, test.values


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s]: %(message)s')

    train, test = read_jester_data()  # read a subset of jester data

    if args.method.startswith('pmf'):
        # Perform mean value imputation on train set for PMF use.
        orig_train = train.copy()         # save a copy for eval later
        train[np.isnan(train)] = train[~np.isnan(train)].mean()

        pmf = PMF(data=train, scale=(-10, 10), dim=args.dimension)
        pmf.build_model()
        pmf.find_map()

        # Check RMSE for MAP estimate
        print 'pmf-map train RMSE: %.5f' % pmf.map_rmse(orig_train)
        print 'pmf-map test RMSE:  %.5f' % pmf.map_rmse(test)

        if args.method == 'pmf-map':
            sys.exit(0)

        # Perform MCMC sampling -- may take a while.
        pmf.sample(n=args.samples, njobs=args.njobs, progressbar=args.verbose)
        pmf.predict(burn_in=args.burn_in)
        print 'pmf-mcmc train rmse: %.5f' % pmf.rmse(orig_train)
        print 'pmf-mcmc test rmse:  %.5f' % pmf.rmse(test)

    elif args.method == 'ur':
        ur_base = UniformRandomBaseline(train)
        print 'uniform random baseline RMSE: %.5f' % ur_base.rmse(test)

    elif args.method == 'gm':
        gm_base = GlobalMeanBaseline(train)
        print 'global mean baseline RMSE: %.5f' % gm_base.rmse(test)

    elif args.method == 'mom':
        mom_base = MeanOfMeansBaseline(train)
        print 'mean of means baseline RMSE: %.5f' % mom_base.rmse(test)

