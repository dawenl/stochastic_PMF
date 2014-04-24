# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import functools
import os
import numpy as np

import pnmf

# <codecell>

specshow = functools.partial(imshow, cmap=cm.PuOr_r, aspect='auto', interpolation='nearest')

# <codecell>

train_tracks = list()
with open('train_tracks.txt', 'rb') as f:
    for line in f:
        train_tracks.append(line.strip())
        
test_tracks = list()
with open('test_tracks.txt', 'rb') as f:
    for line in f:
        test_tracks.append(line.strip())
        
tags = list()
with open('voc.txt', 'rb') as f:
    for line in f:
        tags.append(line.strip())

# <headingcell level=1>

# Batch inference on 10K subset

# <codecell>

K = 1024

np.random.seed(98765)
train_tracks_subset = np.random.choice(train_tracks, size=10000, replace=False)

# <codecell>

D = K + len(tags)

X = np.empty((10000, D), dtype=np.int16)

for (i, tid) in enumerate(train_tracks_subset):
    tdir = os.path.join('vq_hist', '/'.join(tid[2:5]))
    vq = np.load(os.path.join(tdir, '%s_K%d.npy' % (tid, K))).ravel()
    bot = np.load(os.path.join(tdir, '%s_BoT.npy' % tid))
    X[i] = np.hstack((vq, bot))

# <codecell>

specshow(X[:, :K])
colorbar()

# <codecell>

# take tracks with at least 20 tags from test set
X_test = None

for tid in test_tracks:
    tdir = os.path.join('vq_hist', '/'.join(tid[2:5]))
    bot = np.load(os.path.join(tdir, '%s_BoT.npy' % tid))
    if (bot > 0).sum() >= 20:
        vq = np.load(os.path.join(tdir, '%s_K%d.npy' % (tid, K))).ravel()
        if X_test is None:
            X_test = np.hstack((vq, bot))
        else:
            X_test = np.vstack((X_test, np.hstack((vq, bot))))
pass

# <codecell>

hist(np.sum( (X_test[:, K:] > 0), axis=1), bins=50)
pass

# <codecell>

n_components = 128
coder = pnmf.PoissonNMF(n_components=n_components, random_state=98765, verbose=True)

# <codecell>

coder.fit(X)

# <codecell>

# plot 10 "topics"
indices = np.random.choice(n_components, size=10, replace=False)
figure(figsize=(16, 15))
for i in xrange(10):
    subplot(5, 2, i+1)
    bar(np.arange(D), coder.Eb[indices[i]])
    title('Component #%d' % indices[i])

# <codecell>

tagger = pnmf.PoissonNMF(n_components=n_components, random_state=98765, verbose=True)

# <codecell>

tagger.set_components(coder.gamma_b[:, :K], coder.rho_b)

# <codecell>

Et = tagger.transform(X_test[:, :K])

# <codecell>

tags_predicted = Et.dot(coder.Eb[:, K:])

# <codecell>

tags_predicted.max()

# <codecell>

figure(figsize=(16, 8))
tmp = tags_predicted.copy()
subplot(211)
specshow((tmp >= 1), cmap=cm.gray)
colorbar()
subplot(212)
specshow((X_test[:, K:] > 0), cmap=cm.gray)
colorbar()
pass

# <headingcell level=1>

# Stochastic inference on 10K subset

# <codecell>

online_coder = pnmf.OnlinePoissonNMF()

# <headingcell level=1>

# Stochastic inference on the full set

# <codecell>


