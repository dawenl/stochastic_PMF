"""

CREATED: 2014-03-25 02:06:52 by Dawen Liang <dliang@ee.columbia.edu>

"""

import numpy as np
from scipy import special

from sklearn.base import BaseEstimator, TransformerMixin


class PoissonNMF(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=100, max_iter=100, tol=0.0005,
                 smoothness=100, random_state=None, verbose=False,
                 **kwargs):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.setstate(self.random_state)

        self._parse_args(**kwargs)

    def _parse_args(self, **kwargs):
        self.a = float(kwargs.get('a', 0.1))
        self.b = float(kwargs.get('b', 0.1))

    def _init_params(self, n_samples, n_feats):
        # variational parameters for theta
        self.gamma_t = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(n_samples, self.n_components))
        self.rho_t = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(n_samples, self.n_components))
        self.Et, self.Elogt = _compute_expectations(self.gamma_t, self.rho_t)
        # variational parameters for beta
        self.gamma_b = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(self.n_components, n_feats))
        self.rho_b = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(self.n_components, n_feats))
        self.Eb, self.Elogb = _compute_expectations(self.gamma_b, self.rho_b)

    def fit(self, X):
        if not hasattr(self, 'components_'):
            self._init_components(X)
        self._update(X)
        return self

    def transform(self, X, attr=None):
        if attr is None:
            attr = 'Et'
        self._update(X, update_beta=False)
        return getattr(self, attr)

    def _update(self, X, update_beta=True):
        old_bd = -np.inf
        for i in xrange(self.max_iter):
            self._update_theta(X)
            if update_beta:
                self._update_beta(X)
            bound = self._bound(X)
            improvement = (bound - old_bd) / abs(old_bd)
            if self.verbose:
                print('After ITERATION: %d\tObjective: %.2f\t'
                      'Old objective: %.2f\t'
                      'Improvement: %.5f' % (i, bound, old_bd, improvement))
            if improvement < self.tol:
                break
            old_bd = bound
        pass

    def _update_theta(self, X):
        #idx = (X > 0)
        self.c = 1. / np.mean(self.Et)

        xxelinv = X / self._xexplog()
        self.gamma_t = self.a + np.exp(self.Elogt) * np.dot(xxelinv, np.exp(self.Elogb).T)
        self.rho_t = self.a * self.c + np.sum(self.Eb, axis=1)
        self.Et, self.Elogt = _compute_expectations(self.gamma_t, self.rho_t)

    def _update_beta(self, X):
        #idx = (X > 0)
        xxelinv = X / self._xexplog()
        self.gamma_b = self.b + np.exp(self.Elogb) * np.dot(np.exp(self.Elogt).T, xxelinv)
        self.rho_b = self.b + np.sum(self.Et, axis=0)
        self.Eb, self.Elogb = _compute_expectations(self.gamma_b, self.rho_b)

    def _xexplog(self):
        '''
        sum_k exp(E[log theta_{ik} * beta_{kd}])
        '''
        return np.dot(np.exp(self.Elogt), np.exp(self.Elogb))


    def _bound(self, X):
        bound =


def _compute_expectations(alpha, beta):
    '''
    Given x ~ Gam(alpha, beta), compute E[x] and E[log x]
    '''
    return (alpha / beta, special.psi(alpha) - np.log(beta))

class OnlinePoissonNMF():
    pass
