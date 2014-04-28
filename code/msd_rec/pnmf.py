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
        self.smoothness = smoothness
        self.random_state = random_state
        self.verbose = verbose

        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.setstate(self.random_state)

        self._parse_args(**kwargs)

    def _parse_args(self, **kwargs):
        self.a = float(kwargs.get('a', 0.1))
        self.b = float(kwargs.get('b', 0.1))

    def _init_components(self, n_feats):
       # variational parameters for beta
        self.gamma_b = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(self.n_components, n_feats))
        self.rho_b = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(self.n_components, n_feats))
        self.Eb, self.Elogb = _compute_expectations(self.gamma_b, self.rho_b)

    def set_components(self, shape, rate):
        self.gamma_b, self.rho_b = shape, rate
        self.Eb, self.Elogb = _compute_expectations(self.gamma_b, self.rho_b)

    def _init_weights(self, n_samples):
        # variational parameters for theta
        self.gamma_t = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(n_samples, self.n_components))
        self.rho_t = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(n_samples, self.n_components))
        self.Et, self.Elogt = _compute_expectations(self.gamma_t, self.rho_t)
        self.c = 1. / np.mean(self.Et)

    def fit(self, X):
        n_samples, n_feats = X.shape
        self._init_components(n_feats)
        self._init_weights(n_samples)
        self._update(X)
        return self

    def transform(self, X, attr=None):
        if not hasattr(self, 'Eb'):
            raise ValueError('There are no pre-trained components.')
        n_samples, n_feats = X.shape
        if n_feats != self.Eb.shape[1]:
            raise ValueError('The dimension of the transformed data '
                             'does not match with the existing components.')
        if attr is None:
            attr = 'Et'
        self._init_weights(n_samples)
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
        ratio = X / self._xexplog()
        self.gamma_t = self.a + np.exp(self.Elogt) * np.dot(ratio, np.exp(self.Elogb).T)
        self.rho_t = self.a * self.c + np.sum(self.Eb, axis=1)
        self.Et, self.Elogt = _compute_expectations(self.gamma_t, self.rho_t)
        self.c = 1. / np.mean(self.Et)

    def _update_beta(self, X):
        ratio = X / self._xexplog()
        self.gamma_b = self.b + np.exp(self.Elogb) * np.dot(np.exp(self.Elogt).T, ratio)
        self.rho_b = self.b + np.sum(self.Et, axis=0, keepdims=True).T
        self.Eb, self.Elogb = _compute_expectations(self.gamma_b, self.rho_b)

    def _xexplog(self):
        '''
        sum_k exp(E[log theta_{ik} * beta_{kd}])
        '''
        return np.dot(np.exp(self.Elogt), np.exp(self.Elogb))

    def _bound(self, X):
        bound = np.sum(X * np.log(self._xexplog()) - self.Et.dot(self.Eb))
        bound += np.sum((self.a - self.gamma_t) * self.Elogt -
                        (self.a * self.c - self.rho_t) * self.Et +
                        (special.gammaln(self.gamma_t) -
                         self.gamma_t * np.log(self.rho_t)))
        bound += self.n_components * X.shape[0] * self.a * np.log(self.c)
        bound += np.sum((self.b - self.gamma_b) * self.Elogb -
                        (self.b - self.rho_b) * self.Eb +
                        (special.gammaln(self.gamma_b) -
                         self.gamma_b * np.log(self.rho_b)))
        return bound

def _compute_expectations(alpha, beta):
    '''
    Given x ~ Gam(alpha, beta), compute E[x] and E[log x]
    '''
    return (alpha / beta, special.psi(alpha) - np.log(beta))


class OnlinePoissonNMF(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=100, batch_size=10, smoothness=100,
                 max_iter=10, shuffle=True,
                 random_state=None, verbose=False,
                 **kwargs):
        self.n_components = n_components
        self.smoothness = smoothness
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose

        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.setstate(self.random_state)

        self._parse_args(**kwargs)

    def _parse_args(self, **kwargs):
        self.a = float(kwargs.get('a', 0.1))
        self.b = float(kwargs.get('b', 0.1))
        self.t0 = float(kwargs.get('t0', 1.))
        self.kappa = float(kwargs.get('kappa', 0.6))

    def _init_components(self, n_feats):
       # variational parameters for beta
        self.gamma_b = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(self.n_components, n_feats))
        self.rho_b = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(self.n_components, n_feats))
        self.Eb, self.Elogb = _compute_expectations(self.gamma_b, self.rho_b)

    def set_components(self, shape, rate):
        self.gamma_b, self.rho_b = shape, rate
        self.Eb, self.Elogb = _compute_expectations(self.gamma_b, self.rho_b)

    def _init_weights(self, n_samples):
        # variational parameters for theta
        self.gamma_t = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(n_samples, self.n_components))
        self.rho_t = self.smoothness \
            * np.random.gamma(self.smoothness, 1. / self.smoothness,
                              size=(n_samples, self.n_components))
        self.Et, self.Elogt = _compute_expectations(self.gamma_t, self.rho_t)

    def fit(self, X):
        n_samples, n_feats = X.shape
        self.scale = float(n_samples) / self.batch_size
        self._init_components(n_feats)
        self.bound = list()
        for _ in xrange(self.max_iter):
            indices = np.arange(n_samples)
            if self.shuffle:
                np.random.shuffle(indices)
            for i in xrange(0, n_samples, self.batch_size):
                iend = min(i + self.batch_size, n_samples)
                self.rho = (i + self.t0)**(-self.kappa)
                mini_batch = X[indices[i]: indices[iend]]
                self.partial_fit(mini_batch)
                self.bound.append(self._bound(mini_batch))
        return self

    def partial_fit(self, X):
        self.transform(X)
        # take a (natural) gradient step
        ratio = X / self._xexplog()
        self.gamma_b = (1 - self.rho) * self.gamma_b + self.rho * (self.b + self.scale * np.exp(self.Elogb) * np.dot(np.exp(self.Elogt).T, ratio))
        self.rho_b = (1 - self.rho) * self.rho_b + self.rho * (self.b + self.scale * np.sum(self.Et, axis=0, keepdims=True).T)
        self.Eb, self.Elogb = _compute_expectations(self.gamma_b, self.rho_b)


    def transform(self, X, attr=None):
        if not hasattr(self, 'Eb'):
            raise ValueError('There are no pre-trained components.')
        n_samples, n_feats = X.shape
        if n_feats != self.Eb.shape[1]:
            raise ValueError('The dimension of the transformed data '
                             'does not match with the existing components.')
        if attr is None:
            attr = 'Et'
        self._init_weights(n_samples)
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
        self.c = 1. / np.mean(self.Et)

        ratio = X / self._xexplog()
        self.gamma_t = self.a + np.exp(self.Elogt) * np.dot(ratio, np.exp(self.Elogb).T)
        self.rho_t = self.a * self.c + np.sum(self.Eb, axis=1)
        self.Et, self.Elogt = _compute_expectations(self.gamma_t, self.rho_t)

    def _update_beta(self, X):
        ratio = X / self._xexplog()
        self.gamma_b = self.b + np.exp(self.Elogb) * np.dot(np.exp(self.Elogt).T, ratio)
        self.rho_b = self.b + np.sum(self.Et, axis=0, keepdims=True).T
        self.Eb, self.Elogb = _compute_expectations(self.gamma_b, self.rho_b)

    def _xexplog(self):
        '''
        sum_k exp(E[log theta_{ik} * beta_{kd}])
        '''
        return np.dot(np.exp(self.Elogt), np.exp(self.Elogb))

    def _bound(self, X):
        bound = self.scale * np.sum(X * np.log(self._xexplog()) - self.Et.dot(self.Eb))
        bound += self.scale * np.sum((self.a - self.gamma_t) * self.Elogt -
                                     (self.a * self.c - self.rho_t) * self.Et +
                                     (special.gammaln(self.gamma_t) -
                                      self.gamma_t * np.log(self.rho_t)))
        bound += self.scale * self.n_components * X.shape[0] * self.a * np.log(self.c)
        bound += np.sum((self.b - self.gamma_b) * self.Elogb -
                        (self.b - self.rho_b) * self.Eb +
                        (special.gammaln(self.gamma_b) -
                         self.gamma_b * np.log(self.rho_b)))
        return bound

