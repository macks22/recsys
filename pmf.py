import logging
import argparse

import theano
import pymc3 as pm
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as stats


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
        self.alpha = np.ones((n,m)) * alpha

    @property
    def model(self):
        try:
            return self._model
        except:
            self.build_model()
            return self._model

    def build_model(self):
        """Construct the model using pymc3 (most of the work is done by theano).
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

    def sample(self, n=100, njobs=4):
        """Draw n MCMC samples using the NUTS sampler."""
        with self.model:
            logging.info(
                'performing %d MCMC samples using %d jobs' % (n, njobs))
            self.step = pm.NUTS(scaling=self.map)
            self.trace = pm.sample(n, self.step, start=self.map, njobs=njobs)

    def predict(self, burn_in=0):
        n, m = self.data.shape
        std = np.sqrt(1 / self.alpha[0,0])
        self.predicted = np.ndarray((n,m))
        for sample in self.trace[burn_in:]:
            U = sample['U']
            V = sample['V']
            R = np.dot(U, V.T)
            sample_R = np.ndarray((n,m))
            for i in xrange(n):
                for j in xrange(m):
                    sample_R[i,j] = stats.norm.rvs(R[i,j], std)

            self.predicted += sample_R

        self.predicted /= len(self.trace)
        low, high = self.scale
        self.predicted[self.predicted < low] = low
        self.predicted[self.predicted > high] = high

    def rmse(self, data):
        I = ~np.isnan(data)    # indicator for missing values
        N = I.sum()            # number of non-missing values
        sqerror = abs(data - self.predicted) ** 2
        mse = sqerror[I].sum() / N
        return np.sqrt(mse)


def make_parser():
    parser = argparse.ArgumentParser(
        description='Probabilistic Matrix Factorization.')
    parser.add_argument(
        '-n', '--njobs', type=int, default=4,
        help='number of processes to use for MCMC sampling')
    parser.add_argument(
        '-s', '--nsamples', type=int, default=100,
        help='number of MCMC samples to draw')
    parser.add_argument(
        '-b', '--burn-in', type=int, default=10,
        help='number of samples to discard as burn-in')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='enable verbose logging')
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s]: %(message)s')

    logging.info('reading data')
    data = pd.read_csv('data/jester-dataset-v1-dense-first-1000.csv')
    n, m = 50, 20  # subset dimensions
    data = data.head(n).ix[:,:m]  # get subset for model validation

    logging.info('splitting train/test sets')
    n, m = data.shape           # # users, # jokes
    test_size = m / 10          # use 10% of data as test set
    train_size = m - test_size  # and remainder for training

    train = data.copy()
    train.ix[:,train_size:] = np.nan             # remove test data
    train[train.isnull()] = train.mean().mean()  # mean value imputation
    train = train.values

    test = data.copy()
    test.ix[:,:train_size] = np.nan  # remove train data from test set
    test = test.values

    pmf = PMF(data=train, scale=(-10, 10))
    pmf.build_model()
    pmf.find_map()
    pmf.sample(n=args.nsamples, njobs=args.njobs)
    pmf.predict(burn_in=args.burn_in)

    orig_train = data.copy()
    orig_train.ix[:,train_size:] = np.nan  # remove test data
    orig_train = orig_train.values

    train_rmse = pmf.rmse(orig_train)
    test_rmse = pmf.rmse(test)

    print 'train rmse: %.5f' % train_rmse
    print 'test rsme:  %.5f' % test_rmse
