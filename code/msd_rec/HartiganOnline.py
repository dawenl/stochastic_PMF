#!/usr/bin/env python

import numpy as np
from sklearn.base import BaseEstimator

class HartiganOnline(BaseEstimator):
    '''Online Hartigan clustering.'''

    def __init__(self, n_clusters=2, max_iter=10, shuffle=True, verbose=False):
        '''Initialize a Hartigan clusterer.

        :parameters:
        - n_clusters : int
            Number of clusters
    
        - max_iter : int 
            Maximum number of passes through the data

        - shuffle : bool
            Shuffle the data between each pass

        - verbose : bool
            Display debugging output?

        :variables:
        - cluster_centers_ : ndarray, shape=(n_clusters, d)
            Estimated cluster centroids

        - cluster_sizes_ : ndarray, shape=(n_clusters)
            Size (number of points) for each cluster
        '''

        self.n_clusters     = n_clusters
        self.max_iter       = max_iter
        self.shuffle        = shuffle
        self.verbose        = verbose

        self.cluster_sizes_     = np.zeros(self.n_clusters)

    def fit(self, X):
        '''Fit the cluster centers.

        :parameters:
        - X : ndarray, size=(n, d)
            The data to be clustered
        '''

        n, d = X.shape

        # Initialize the cluster centers, costs, sizes
        self.cluster_centers_   = np.zeros( (self.n_clusters, d), dtype=X.dtype)

        step = 0

        idx = np.arange(n)
        while step < self.max_iter:
            step = step + 1

            # Should we shuffle the data?
            if self.shuffle:
                np.random.shuffle(idx)

            self.partial_fit(X[idx])

    def partial_fit(self, X):
        '''Partial fit the cluster centers'''

        n, d = X.shape

        if not hasattr(self, 'cluster_centers_'):
            self.cluster_centers_   = np.zeros( (self.n_clusters, d), dtype=X.dtype)
            
        balances = self.cluster_sizes_ / (1.0 + self.cluster_sizes_) 
        norms    = np.sum(self.cluster_centers_**2, axis=1)

        for xi in X:
            # Get the closest cluster center
            j = np.argmin(balances * (np.sum(xi**2) + norms - 2 * self.cluster_centers_.dot(xi)))

            # Update the center
            self.cluster_centers_[j] = (self.cluster_sizes_[j] * self.cluster_centers_[j] + xi) / (1.0 + self.cluster_sizes_[j])

            # Update the counter
            self.cluster_sizes_[j] += 1.0

            # Update the balance
            balances[j] = self.cluster_sizes_[j] / (1.0 + self.cluster_sizes_[j])

            # Update the norms
            norms[j] = np.sum(self.cluster_centers_[j]**2)
