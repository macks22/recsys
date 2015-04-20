import sys
import logging
import argparse

import theano
import pymc3 as pm
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt


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


def rmse(test_data, predicted):
    """Calculate root mean squared error, ignoring missing values in the test
    data."""
    I = ~np.isnan(test_data)   # indicator for missing values
    N = I.sum()                # number of non-missing values
    sqerror = abs(test_data - predicted) ** 2
    mse = sqerror[I].sum() / N
    return np.sqrt(mse)


class Model(object):
    """Base class for PyMC model wrappers."""

    def __init__(self):
        # Set None values for attributes to be set later
        self.step = None
        self.trace = None
        self.predicted = None

    @property
    def model(self):
        try:
            return self._model
        except:
            self.build_model()
            return self._model

    def build_model(self):
        raise NotImplementedError()

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

    @property
    def start(self):
        return self.map

    def sample(self, n=100, njobs=4, progressbar=True):
        """Draw n MCMC samples using the NUTS sampler."""
        with self.model:
            logging.info(
                'drawing %d MCMC samples using %d jobs' % (n, njobs))
            self.step = pm.NUTS(scaling=self.start)
            self.trace = pm.sample(n, self.step, start=self.start, njobs=njobs,
                                   progressbar=progressbar)

    def rmse(self, test_data):
        """Find root mean squared error on this model's predictions."""
        return rmse(test_data, self.predicted)


class PMF(Model):
    """Probabilistic Matrix Factorization"""

    def __init__(self, data, scale, alpha=2, dim=10):
        """
        :param numpy.ndarray data: Observed data to learn the model from.
        :param int alpha: Precision for all observed data.
        :param int dim: Dimensionality, or number of latent factors to learn.
        """
        super(PMF, self).__init__()
        self.scale = scale
        self.dim = dim
        self.data = data.copy()

        # Mean value imputation
        nan_mask = np.isnan(self.data)
        self.data[nan_mask] = self.data[~nan_mask].mean()

        # Low precision reflects uncertainty; prevents overfitting
        # Set to mean variance across users and items.
        self.alpha_u = 1 / train.var(axis=1).mean()
        self.alpha_v = 1 / train.var(axis=0).mean()

        # Use fixed precision for the likelihood function.
        self.alpha = alpha

    @property
    def std(self):
        return np.sqrt(1 / self.alpha)

    def build_model(self, std=0.5):
        """Construct the model using pymc3 (most of the work is done by theano).
        Note that the `testval` param for U and V initialize the model away from
        0 using a small amount of Gaussian noise, set by `std`.
        """
        n, m = self.data.shape
        dim = self.dim

        logging.info('building the PMF model')
        with pm.Model() as pmf:
            U = pm.MvNormal(
                'U', mu=0, tau=self.alpha_u,
                shape=(n, dim), testval=np.random.randn(n, dim) * std)
            V = pm.MvNormal(
                'V', mu=0, tau=self.alpha_v,
                shape=(m, dim), testval=np.random.randn(m, dim) * std)
            R = pm.Normal(
                'R', mu=theano.tensor.dot(U, V.T),
                tau=self.alpha * np.ones(self.data.shape),
                observed=self.data)

        logging.info('done building PMF model')
        self._model = pmf

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
        """Fill in missing values in data using MCMC samples to approximate the
        posterior distribution. Discard `burn_in` samples and use the rest.
        Cache the predictions on the object and also return them.
        """
        self.predicted = np.ndarray(self.data.shape)
        for sample in self.trace[burn_in:]:
            self.predicted += self.estimate_R(sample['U'], sample['V'])

        self.predicted /= len(self.trace[burn_in:])
        low, high = self.scale
        self.predicted[self.predicted < low] = low
        self.predicted[self.predicted > high] = high
        return self.predicted

    def running_rmse(self, test_data, burn_in=10, plot=True):
        """Calculate RMSE for each step of the trace to monitor convergence.
        Return a list of tuples with the first element being the per-sample
        RMSE, and the last being the running RMSE.
        """
        burn_in = burn_in if len(self.trace) >= burn_in else 0
        results = {'per-step': [], 'running': []}
        R = np.zeros(self.data.shape)
        for cnt, sample in enumerate(self.trace[burn_in:]):
            sample_R = self.estimate_R(sample['U'], sample['V'])
            R += sample_R
            running_R = R / (cnt + 1)
            results['per-step'].append(rmse(test_data, sample_R))
            results['running'].append(rmse(test_data, running_R))

        # Plot the results before returning
        results = pd.DataFrame(results)
        results.plot(
            kind='line', grid=False, figsize=(16, 7),
            title="Posterior Predictive Per-step and Running RMSE")

        # Return the final predictions, and the RMSE calculations
        return running_R, pd.DataFrame(results)

    def _norms(self, ord, monitor):
        logging.info('calculating %s norms for model vars: %s' % (
            ord, ','.join(monitor)))

        norms = {var: [] for var in monitor}
        for sample in self.trace:
            for var in monitor:
                norms[var].append(np.linalg.norm(sample[var], ord))
        return norms

    def norms(self, ord='fro'):
        """Return norms of latent variables. These can be used to monitor
        convergence of the sampler.
        """
        monitor = ('U', 'V')
        return self._norms(ord, monitor)

    def traceplot(self):
        """Plot Frobenius norms of all variables in the trace."""
        trace_norms = self.norms()
        num_plots = len(trace_norms)
        num_rows = int(np.ceil(num_plots / 2.))
        fig, axes = plt.subplots(num_rows, 2)
        for key, ax in zip(trace_norms, axes):
            title = '$\|%s\|_{Fro}^2$ at Each Sample' % (
                key if len(key) == 1 else '\%s' % key)
            series = pd.Series(trace_norms[key])
            series.plot(kind='line', grid=False, title=title, ax=ax)
        fig.show()


class BPMF(PMF):
    """Bayesian Probabilistic Matrix Factorization"""

    def __init__(self, data, scale, alpha=2, dim=10):
        """
        :param numpy.ndarray data: Observed data to learn the model from.
        :param (tuple of int) scale: scale of user preferences (R).
        :param int dim: Dimensionality, or number of latent factors to learn.
        """
        super(BPMF, self).__init__(data, scale, alpha, dim)

        # BPMF uses hyperpriors to learn the precision vectors.
        del self.__dict__['alpha_u']
        del self.__dict__['alpha_v']

    def build_model(self, std=0.5):
        n, m = self.data.shape
        dim = self.dim
        beta_0 = 1  # scaling factor for lambdas; unclear on its use

        logging.info('building the BPMF model')
        with pm.Model() as bpmf:
            # Specify user feature matrix
            lambda_u = pm.Wishart(
                'lambda_u', n=dim, V=np.eye(dim), shape=(dim, dim),
                testval=np.random.randn(dim, dim) * std)
            mu_u = pm.Normal(
                'mu_u', mu=0, tau=beta_0 * lambda_u, shape=dim,
                 testval=np.random.randn(dim) * std)
            U = pm.MvNormal(
                'U', mu=mu_u, tau=lambda_u, shape=(n, dim),
                testval=np.random.randn(n, dim) * std)

            # Specify item feature matrix
            lambda_v = pm.Wishart(
                'lambda_v', n=dim, V=np.eye(dim), shape=(dim, dim),
                testval=np.random.randn(dim, dim) * std)
            mu_v = pm.Normal(
                'mu_v', mu=0, tau=beta_0 * lambda_v, shape=dim,
                 testval=np.random.randn(dim) * std)
            V = pm.MvNormal(
                'V', mu=mu_v, tau=lambda_v, shape=(m, dim),
                testval=np.random.randn(m, dim) * std)

            # Specify rating likelihood function
            R = pm.Normal(
                'R', mu=theano.tensor.dot(U, V.T), tau=self.alpha,
                observed=self.data)

        logging.info('done building the BPMF model')
        self._model = bpmf

    def find_map(self):
        """Use the PMF map to initialize BPMF."""
        logging.info("Using PMF MAP to initialize BPMF")
        pmf = PMF(self.data, self.scale, dim=self.dim)
        self._map = pmf.map

    @property
    def start(self):
        """Initialization values for the sampler."""
        start = self.map
        point = self.model.test_point
        for key in point:
            if key not in start:
                start[key] = point[key]
        return start

    def norms(self, ord='fro'):
        """Return norms of latent variables. These can be used to monitor
        convergence of the sampler.
        """
        monitor = ('U', 'V', 'lambda_u', 'lambda_v')
        return self._norms(ord, monitor)


class Baseline(object):
    """Calculate baseline predictions."""

    def __init__(self, train_data):
        """Simple heuristic-based transductive learning to fill in missing
        values in data matrix."""
        self.predict(train_data.copy())

    def predict(self, train_data):
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
    """Fill in missing values using mean of user/item/global means."""

    def predict(self, train_data):
        nan_mask = np.isnan(train_data)
        masked_train = np.ma.masked_array(train_data, nan_mask)
        global_mean = masked_train.mean()
        user_means = masked_train.mean(axis=1)
        item_means = masked_train.mean(axis=0)
        self.predicted = train_data.copy()
        n, m = train_data.shape
        for i in xrange(n):
            for j in xrange(m):
                if np.ma.isMA(item_means[j]):
                    self.predicted[i,j] = np.mean(
                        (global_mean, user_means[i]))
                else:
                    self.predicted[i,j] = np.mean(
                        (global_mean, user_means[i], item_means[j]))


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
        '-m', '--method',
        choices=('ur', 'gm', 'mom', 'pmf-map', 'pmf-mcmc', 'bpmf'),
        default='pmf-map',
        help='method to use for making predictions')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='enable verbose logging')
    return parser


def read_jester_data(n=1000, m=100):
    """Read a subset of the dense 1000x100 jester dataset.
    :param int n: Number of users to keep.
    :param int m: Number of jokes to keep.
    """
    logging.info('reading data')
    data = pd.read_csv('data/jester-dataset-v1-dense-first-1000.csv')
    data = data.head(n).ix[:,:m]  # get subset for model validation

    # First we need to split up our data into a training set and a test set.
    logging.info('splitting train/test sets')
    n, m = data.shape           # # users, # jokes
    N = n * m                   # # cells in matrix
    test_size = N / 10          # use 10% of data as test set
    train_size = N - test_size  # and remainder for training

    # Prepare train/test ndarrays.
    train = data.copy().values
    test = np.ones(data.shape) * np.nan

    # Draw random sample of training data to use for testing.
    tosample = np.where(~np.isnan(train))
    idx_pairs = zip(tosample[0], tosample[1])
    indices = np.arange(len(idx_pairs))
    sample = np.random.choice(indices, replace=False, size=test_size)

    # Transfer random sample from train set to test set.
    for idx in sample:
        idx_pair = idx_pairs[idx]
        test[idx_pair] = train[idx_pair]  # transfer to test set
        train[idx_pair] = np.nan          # remove from train set

    # Verify everything worked properly
    assert(np.isnan(train).sum() == test_size)
    assert(np.isnan(test).sum() == train_size)

    # Return the two numpy ndarrays
    return train, test


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s]: %(message)s')

    train, test = read_jester_data()  # read a subset of jester data
    ratings_range = (-10, 10)

    if args.method == 'bpmf':
        bpmf = BPMF(data=train, scale=ratings_range, dim=args.dimension)
        bpmf.build_model()
        bpmf.find_map()

        # Check RMSE for MAP estimate (same as PMF MAP.
        print 'bpmf-map train RMSE: %.5f' % bpmf.map_rmse(train)
        print 'bpmf-map test RMSE:  %.5f' % bpmf.map_rmse(test)

        # Perform MCMC sampling -- may take a while.
        bpmf.sample(n=args.samples, njobs=args.njobs, progressbar=args.verbose)
        bpmf.predict(burn_in=args.burn_in)
        print 'pmf-mcmc train rmse: %.5f' % bpmf.rmse(train)
        print 'pmf-mcmc test rmse:  %.5f' % bpmf.rmse(test)

    if args.method.startswith('pmf'):
        pmf = PMF(data=train, scale=ratings_range, dim=args.dimension)
        pmf.build_model()
        pmf.find_map()

        # Check RMSE for MAP estimate
        print 'pmf-map train RMSE: %.5f' % pmf.map_rmse(train)
        print 'pmf-map test RMSE:  %.5f' % pmf.map_rmse(test)

        if args.method == 'pmf-map':
            sys.exit(0)

        # Perform MCMC sampling -- may take a while.
        pmf.sample(n=args.samples, njobs=args.njobs, progressbar=args.verbose)
        pmf.predict(burn_in=args.burn_in)
        print 'pmf-mcmc train rmse: %.5f' % pmf.rmse(train)
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

