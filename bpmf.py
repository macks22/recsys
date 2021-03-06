import os
import sys
import time
import hashlib
import logging
import argparse

try:
    import ujson as json
except ImportError:
    import json

import theano
import theano.tensor as t
import pymc3 as pm
import numpy as np
import pandas as pd
import scipy as sp
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
    return np.sqrt(((test_data - predicted) ** 2)[I].sum() / N)


# We adapt these from `pymc3`'s `backends` module, where the original
# code is used to save the traces from MCMC samples.
def save_np_vars(vars, savedir):
    """Save a dictionary of numpy variables to `savedir`. We assume
    the directory does not exist; an OSError will be raised if it does.
    """
    logging.info('writing numpy vars to directory: %s' % savedir)
    os.mkdir(savedir)
    shapes = {}
    for varname in vars:
        data = vars[varname]
        var_file = os.path.join(savedir, varname + '.txt')
        np.savetxt(var_file, data.reshape(-1, data.size))
        shapes[varname] = data.shape

        ## Store shape information for reloading.
        shape_file = os.path.join(savedir, 'shapes.json')
        with open(shape_file, 'w') as sfh:
            json.dump(shapes, sfh)


def load_np_vars(savedir):
    """Load numpy variables saved with `save_np_vars`."""
    shape_file = os.path.join(savedir, 'shapes.json')
    with open(shape_file, 'r') as sfh:
        shapes = json.load(sfh)

    vars = {}
    for varname, shape in shapes.items():
        var_file = os.path.join(savedir, varname + '.txt')
        vars[varname] = np.loadtxt(var_file).reshape(shape)

    return vars


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

    def find_map(self, verbose=False, savedir=None):
        """Use non-gradient based optimization to find MAP."""
        tstart = time.time()
        with self.model:
            logging.info('finding MAP using Powell optimization...')
            self._map = pm.find_MAP(fmin=sp.optimize.fmin_powell, disp=verbose)

        elapsed = time.time() - tstart
        logging.info('found MAP in %d seconds' % int(elapsed))

        if savedir:
            self.save_map(savedir)

    def save_map(self, savedir):
        logging.info('writing MAP to directory: %s' % savedir)
        save_np_vars(self.map, savedir)

    def load_map(self, savedir):
        self._map = load_np_vars(savedir)

    @property
    def start(self):
        return self.map

    def sample(self, n=100, njobs=4, progressbar=True, start=None, trace=None):
        """Draw n MCMC samples using the NUTS sampler."""
        start = self.start if start is None else start
        with self.model:
            logging.info(
                'drawing %d MCMC samples using %d jobs' % (n, njobs))
            self.step = pm.NUTS(scaling=start)
            self.trace = pm.sample(n, self.step, start=start, njobs=njobs,
                                   progressbar=progressbar, trace=trace)

    def load_trace(self, savedir):
        self.trace = pm.backends.text.load(savedir)

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
        self.alpha_u = 1 / self.data.var(axis=1).mean()
        self.alpha_v = 1 / self.data.var(axis=0).mean()

        # Use fixed precision for the likelihood function.
        self.alpha = alpha

    @property
    def std(self):
        return np.sqrt(1. / self.alpha)

    def build_model(self, std=0.01):
        """Construct the model using pymc3 (most of the work is done by theano).
        Note that the `testval` param for U and V initialize the model away from
        0 using a small amount of Gaussian noise, set by `std`.
        """
        n, m = self.data.shape
        dim = self.dim

        logging.info('building the PMF model')
        with pm.Model() as pmf:
            U = pm.MvNormal(
                'U', mu=0, tau=self.alpha_u * np.eye(dim),
                shape=(n, dim), testval=np.random.randn(n, dim) * std)
            V = pm.MvNormal(
                'V', mu=0, tau=self.alpha_v * np.eye(dim),
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
        sample_R = np.array([
            [np.random.normal(R[i,j], self.std) for j in xrange(m)]
            for i in xrange(n)
        ])

        # Bound predictions based on rating scale.
        low, high = self.scale
        sample_R[sample_R < low] = low
        sample_R[sample_R > high] = high
        return sample_R

    def map_rmse(self, test_data):
        sample_R = self.estimate_R(self.map['U'], self.map['V'])
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
        return self.predicted

    def running_rmse(self, test_data, burn_in=10, plot=False):
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
        return running_R, results

    def _norms(self, monitor):
        logging.info('calculating Frobenius norms for model vars: %s' % (
            ','.join(monitor)))

        norms = {var: [] for var in monitor}
        for sample in self.trace:
            for var in monitor:
                norms[var].append(np.linalg.norm(sample[var]))
        return norms

    def norms(self):
        """Return norms of latent variables. These can be used to monitor
        convergence of the sampler.
        """
        monitor = ('U', 'V')
        return self._norms(monitor)

    def traceplot(self):
        """Plot Frobenius norms of all variables in the trace."""
        trace_norms = self.norms()
        num_plots = len(trace_norms)
        num_rows = int(np.ceil(num_plots / 2.))
        fig, axes = plt.subplots(num_rows, 2)
        for key, ax in zip(trace_norms, axes.flat):
            title = '$\|%s\|_{Fro}^2$ at Each Sample' % key
            series = pd.Series(trace_norms[key])
            series.plot(kind='line', grid=False, title=title, ax=ax)
        fig.show()
        return fig, axes


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

    def build_model(self, std=0.01):
        n, m = self.data.shape
        dim = self.dim
        beta_0 = 1  # scaling factor for lambdas; unclear on its use

        # We will use separate priors for sigma and correlation matrix.
        # In order to convert the upper triangular correlation values to a
        # complete correlation matrix, we need to construct an index matrix:
        n_elem = dim * (dim - 1) / 2
        tri_index = np.zeros([dim, dim], dtype=int)
        tri_index[np.triu_indices(dim, k=1)] = np.arange(n_elem)
        tri_index[np.triu_indices(dim, k=1)[::-1]] = np.arange(n_elem)

        logging.info('building the BPMF model')
        with pm.Model() as bpmf:
            # Specify user feature matrix
            sigma_u = pm.Uniform('sigma_u', shape=dim)
            corr_triangle_u = pm.LKJCorr(
                'corr_u', n=1, p=dim,
                testval=np.random.randn(n_elem) * std)

            corr_matrix_u = corr_triangle_u[tri_index]
            corr_matrix_u = t.fill_diagonal(corr_matrix_u, 1)
            cov_matrix_u = t.diag(sigma_u).dot(corr_matrix_u.dot(t.diag(sigma_u)))
            lambda_u = t.nlinalg.matrix_inverse(cov_matrix_u)

            mu_u = pm.Normal(
                'mu_u', mu=0, tau=beta_0 * t.diag(lambda_u), shape=dim,
                 testval=np.random.randn(dim) * std)
            U = pm.MvNormal(
                'U', mu=mu_u, tau=lambda_u,
                shape=(n, dim), testval=np.random.randn(n, dim) * std)

            # Specify item feature matrix
            sigma_v = pm.Uniform('sigma_v', shape=dim)
            corr_triangle_v = pm.LKJCorr(
                'corr_v', n=1, p=dim,
                testval=np.random.randn(n_elem) * std)

            corr_matrix_v = corr_triangle_v[tri_index]
            corr_matrix_v = t.fill_diagonal(corr_matrix_v, 1)
            cov_matrix_v = t.diag(sigma_v).dot(corr_matrix_v.dot(t.diag(sigma_v)))
            lambda_v = t.nlinalg.matrix_inverse(cov_matrix_v)

            mu_v = pm.Normal(
                'mu_v', mu=0, tau=beta_0 * t.diag(lambda_v), shape=dim,
                 testval=np.random.randn(dim) * std)
            V = pm.MvNormal(
                'V', mu=mu_v, tau=lambda_v,
                shape=(m, dim), testval=np.random.randn(m, dim) * std)

            # Specify rating likelihood function
            R = pm.Normal(
                'R', mu=t.dot(U, V.T), tau=self.alpha, observed=self.data)

        logging.info('done building the BPMF model')
        self._model = bpmf

    def find_map(self, verbose=False, savedir=None):
        """Use the PMF map to initialize BPMF."""
        logging.info("Using PMF MAP to initialize BPMF")
        pmf = PMF(self.data, self.scale, dim=self.dim)
        pmf.find_map(verbose, savedir)
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

    def norms(self):
        """Return norms of latent variables. These can be used to monitor
        convergence of the sampler.
        """
        monitor = ('U', 'V', 'corr_u', 'corr_v', 'mu_u', 'mu_v')
        return self._norms(monitor)


class Baseline(object):
    """Calculate baseline predictions."""

    def __init__(self, data):
        """Simple heuristic-based transductive learning to fill in missing
        values in data matrix."""
        self.predict(data.copy())

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
        dmin, dmax = masked_train.min(), masked_train.max()
        N = nan_mask.sum()
        train_data[nan_mask] = np.random.uniform(dmin, dmax, N)
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
        '-nj', '--njobs', type=int, default=4,
        help='number of processes to use for MCMC sampling')
    parser.add_argument(
        '-ns', '--nsamples', type=int, default=100,
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
        '-st', '--save-trace', default=None,
        help='directory to save MCMC trace to')
    parser.add_argument(
        '-lt', '--load-trace', default=None,
        help='directory to load MCMC trace from')
    parser.add_argument(
        '-sm', '--save-map', default=None,
        help='directory to save the MAP estimate to for {B}PMF')
    parser.add_argument(
        '-lm', '--load-map', default=None,
        help='directory to load MAP estimate from for {B}PMF')
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

    # Finally, hash the indices and save the train/test sets.
    index_string = ''.join(map(str, np.sort(sample)))
    name = hashlib.sha1(index_string).hexdigest()
    savedir = os.path.join('data', name)
    save_np_vars({'train': train, 'test': test}, savedir)

    # Return the two numpy ndarrays
    return train, test, name


def load_train_test(name):
    """Load the train/test sets."""
    savedir = os.path.join('data', name)
    vars = load_np_vars(savedir)
    return vars['train'], vars['test']


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s]: %(message)s')

    train, test, name = read_jester_data()  # read a subset of jester data
    logging.info('saved train/test split to %s:' % name)
    ratings_range = (-10, 10)

    if args.method == 'bpmf':
        bpmf = BPMF(data=train, scale=ratings_range, dim=args.dimension)
        bpmf.build_model()
        if args.load_map:
            bpmf.load_map(args.load_map)
        else:
            bpmf.find_map(verbose=args.verbose, savedir=args.save_map)

        # Check RMSE for MAP estimate (same as PMF MAP).
        print 'bpmf-map train RMSE: %.5f' % bpmf.map_rmse(train)
        print 'bpmf-map test RMSE:  %.5f' % bpmf.map_rmse(test)

        # Perform MCMC sampling -- may take a while.
        if args.load_trace:
            with bpmf.model:
                backend = pm.backend.text.load(args.load_trace)
        elif args.save_trace:
            with bpmf.model:
                backend = pm.backends.Text(args.save_trace)
        else:
            backend = None

        bpmf.sample(n=args.nsamples, njobs=args.njobs,
                    progressbar=args.verbose,
                    trace=backend)
        bpmf.predict(burn_in=args.burn_in)
        print 'bpmf-mcmc train rmse: %.5f' % bpmf.rmse(train)
        print 'bpmf-mcmc test rmse:  %.5f' % bpmf.rmse(test)

    elif args.method.startswith('pmf'):
        pmf = PMF(data=train, scale=ratings_range, dim=args.dimension)
        pmf.build_model()
        if args.load_map:
            pmf.load_map(args.load_map)
        else:
            pmf.find_map(verbose=args.verbose, savedir=args.save_map)

        # Check RMSE for MAP estimate
        print 'pmf-map train RMSE: %.5f' % pmf.map_rmse(train)
        print 'pmf-map test RMSE:  %.5f' % pmf.map_rmse(test)

        if args.method == 'pmf-map':
            sys.exit(0)

        # Perform MCMC sampling -- may take a while.
        if args.load_trace:
            with pmf.model:
                backend = pm.backend.text.load(args.load_trace)
        elif args.save_trace:
            with pmf.model:
                backend = pm.backends.Text(args.save_trace)
        else:
            backend = None

        pmf.sample(n=args.nsamples, njobs=args.njobs,
                   progressbar=args.verbose,
                   trace=backend)
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

