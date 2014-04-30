# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import functools
import os
import numpy as np
from joblib import Parallel, delayed

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

# <codecell>

def construct_pred_mask(tags_predicted, predictat):
    n_samples, n_tags = tags_predicted.shape
    rankings = np.argsort(-tags_predicted, axis=1)[:, :predictat]
    tags_predicted_binary = np.zeros_like(tags_predicted, dtype=bool)
    for i in xrange(n_samples):
        tags_predicted_binary[i, rankings[i]] = 1
    return tags_predicted_binary

def per_tag_prec_recall(tags_predicted_binary, tags_true_binary):
    mask = np.logical_and(tags_predicted_binary, tags_true_binary)
    prec = mask.sum(axis=0) / (np.sum(tags_predicted_binary, axis=0) + np.spacing(1))
    recall = mask.sum(axis=0) / (np.sum(tags_true_binary, axis=0) + np.spacing(1))
    return prec, recall

# <codecell>

# take tracks with at least 20 tags 
y_test = None

test_tracks_selected = list()

for tid in test_tracks:
    tdir = os.path.join('vq_hist', '/'.join(tid[2:5]))
    bot = np.load(os.path.join(tdir, '%s_BoT.npy' % tid))
    if (bot > 0).sum() >= 20:
        test_tracks_selected.append(tid)
        if y_test is None:
            y_test = bot
        else:
            y_test = np.vstack((y_test, bot))

# <codecell>

hist(np.sum( (y_test > 0), axis=1), bins=50)
pass

# <headingcell level=1>

# Batch inference on 10K subset

# <codecell>

K = 512

np.random.seed(98765)
train_tracks_subset = np.random.choice(train_tracks, size=10000, replace=False)

# <codecell>

D = K + len(tags)

X = np.empty((10000, D), dtype=np.int16)

for (i, tid) in enumerate(train_tracks_subset):
    tdir = os.path.join('vq_hist', '/'.join(tid[2:5]))
    vq = np.load(os.path.join(tdir, '%s_K%d.npy' % (tid, K))).ravel()
    bot = np.load(os.path.join(tdir, '%s_BoT.npy' % tid))
    bot[bot > 0] = 1
    X[i] = np.hstack((vq, bot))

# <codecell>

bar(np.arange(D), X[1000])

# <codecell>

X_test = np.empty((len(test_tracks_selected), K), dtype=int16)

for (i, tid) in enumerate(test_tracks_selected):
    tdir = os.path.join('vq_hist', '/'.join(tid[2:5]))
    vq = np.load(os.path.join(tdir, '%s_K%d.npy' % (tid, K))).ravel()
    X_test[i] = vq

# <codecell>

reload(pnmf)
n_components = 100
coder = pnmf.PoissonNMF(n_components=n_components, random_state=98765, verbose=True)

# <codecell>

coder.fit(X)

# <codecell>

# randomly plot 30 "topics"
indices = np.random.choice(n_components, size=30, replace=False)
figure(figsize=(45, 15))
for i in xrange(30):
    subplot(10, 3, i+1)
    topic = coder.Eb[indices[i]].copy()
    # properly normalize the BoT dimensions for visualization purposes
    topic[K:] /= topic[K:].max()
    topic[K:] *= topic[:K].max()
    bar(np.arange(D), topic)
    axvline(x=K, color='red')
    title('Component #%d' % indices[i])
#savefig('dict.eps')

# <codecell>

tagger = pnmf.PoissonNMF(n_components=n_components, random_state=98765, verbose=True)

# <codecell>

tagger.set_components(coder.gamma_b[:, :K], coder.rho_b)

# <codecell>

Et = tagger.transform(X_test)

# <codecell>

tags_predicted = Et.dot(coder.Eb[:, K:])
n_samples, n_tags = tags_predicted.shape

print tags_predicted.min(), tags_predicted.max()

div_factor = 5
tags_predicted = tags_predicted - div_factor * np.mean(tags_predicted, axis=0)

# <codecell>

predictat = 20
tags_predicted_binary = construct_pred_mask(tags_predicted, predictat)
tags_true_binary = (y_test > 0)

# <codecell>

prec, recall = per_tag_prec_recall(tags_predicted_binary, tags_true_binary)
print np.mean(prec), np.std(prec) / sqrt(n_tags)
print np.mean(recall), np.std(recall) / sqrt(n_tags)

# <codecell>

tags_random_binary = np.zeros((n_samples, n_tags), dtype=bool)
for i in xrange(n_samples):
    idx = np.random.choice(n_tags, size=predictat, replace=False)
    tags_random_binary[i, idx] = 1

# <codecell>

prec, recall = per_tag_prec_recall(tags_random_binary, tags_true_binary)
print np.mean(prec), np.std(prec) / sqrt(n_tags)
print np.mean(recall), np.std(recall) / sqrt(n_tags)

# <headingcell level=1>

# Stochastic inference on 10K subset

# <codecell>

reload(pnmf)

n_components = 100
online_coder = pnmf.OnlinePoissonNMF(n_components=n_components, batch_size=500, n_pass=10, 
                                     random_state=98765, verbose=True)

# <codecell>

online_coder.fit(X)

# <codecell>

plot(online_coder.bound)
axhline(y=coder._bound(X), color='red')
pass

# <codecell>

# randomly plot 30 "topics"
indices = np.random.choice(n_components, size=30, replace=False)
figure(figsize=(45, 15))
for i in xrange(30):
    subplot(10, 3, i+1)
    topic = online_coder.Eb[indices[i]].copy()
    # properly normalize the BoT dimensions for visualization purposes
    #topic[K:] /= topic[K:].max()
    #topic[K:] *= topic[:K].max()
    bar(np.arange(D), topic)
    axvline(x=K, color='red')
    title('Component #%d' % indices[i])
#savefig('stoc_dict.eps')

# <codecell>

ents = np.zeros((n_components, ))

for k in xrange(n_components):
    ents[k] = scipy.stats.entropy(online_coder.Eb[k])   

# <codecell>

idx = np.argsort(-ents)

# <codecell>

plot(ents[idx], '-o')
pass

# <codecell>

tagger = pnmf.PoissonNMF(n_components=n_components, random_state=98765, verbose=True)

# <codecell>

tagger.set_components(online_coder.gamma_b[:, :K], online_coder.rho_b[:, :K])

# <codecell>

Et = tagger.transform(X_test)

# <codecell>

tags_predicted = Et.dot(online_coder.Eb[:, K:])
n_samples, n_tags = tags_predicted.shape

print tags_predicted.min(), tags_predicted.max()

div_factor = 3
tags_predicted = tags_predicted - div_factor * np.mean(tags_predicted, axis=0)

# <codecell>

predictat = 20
tags_predicted_binary = construct_pred_mask(tags_predicted, predictat)
tags_true_binary = (y_test > 0)

# <codecell>

prec, recall = per_tag_prec_recall(tags_predicted_binary, tags_true_binary)
print np.mean(prec), np.std(prec) / sqrt(n_tags)
print np.mean(recall), np.std(recall) / sqrt(n_tags)

# <headingcell level=1>

# Stochastic inference on the full set

# <codecell>

def ooc_fit(obj, train_tracks, K, n_feats):
    n_samples = (len(train_tracks) / obj.batch_size) * obj.batch_size
    obj.scale = float(n_samples) / obj.batch_size
    obj._init_components(n_feats)
    obj.bound = list()
    for count in xrange(obj.n_pass):
        print 'Iteration %d: passing through the data...' % count
        indices = np.arange(n_samples)
        if obj.shuffle:
            np.random.shuffle(indices)
        for (i, istart) in enumerate(xrange(0, n_samples,
                                            obj.batch_size), 1):
            iend = min(istart + obj.batch_size, n_samples)
            obj.rho = (i + obj.t0)**(-obj.kappa)
            mini_batch = np.zeros((iend - istart, n_feats))
            for s in xrange(iend - istart):
                tid = train_tracks[indices[istart + s]]
                #print '\tRead in track: %s' % tid
                tdir = os.path.join('vq_hist', '/'.join(tid[2:5]))
                vq = np.load(os.path.join(tdir, '%s_K%d.npy' % (tid, K))).ravel()
                bot = np.load(os.path.join(tdir, '%s_BoT.npy' % tid))
                bot[bot > 0] = 1                
                mini_batch[s] = np.hstack((vq, bot))
            obj.partial_fit(mini_batch)
            obj.bound.append(obj._stoch_bound(mini_batch))
            if not i % 50:
                print '\t%d mini-batches finished.' % i
    return obj

# <codecell>

reload(pnmf)

n_components = 100
online_coder_full = pnmf.OnlinePoissonNMF(n_components=n_components, batch_size=1000, n_pass=1, 
                                          random_state=98765, verbose=True)

# <codecell>

K = 2048
D = K + len(tags)

online_coder_full = ooc_fit(online_coder_full, train_tracks, K, D)

# <codecell>

plot(online_coder_full.bound)
pass

# <codecell>

np.mean(online_coder_full.bound[-100:])

# <codecell>

# randomly plot 30 "topics"
indices = np.random.choice(n_components, size=30, replace=False)
figure(figsize=(45, 15))
for i in xrange(30):
    subplot(10, 3, i+1)
    topic = online_coder_full.Eb[indices[i]].copy()
    # properly normalize the BoT dimensions for visualization purposes
    #topic[K:] /= topic[K:].max()
    #topic[K:] *= topic[:K].max()
    bar(np.arange(D), topic)
    axvline(x=K, color='red')
    title('Component #%d' % indices[i])
#savefig('stoc_dict.eps')

# <codecell>

ents = np.zeros((n_components, ))

for k in xrange(n_components):
    ents[k] = scipy.stats.entropy(online_coder_full.Eb[k])   

# <codecell>

idx = np.argsort(-ents)

# <codecell>

plot(ents[idx], '-o')
pass

# <codecell>

tagger = pnmf.PoissonNMF(n_components=n_components, random_state=98765, verbose=True)

# <codecell>

tagger.set_components(online_coder_full.gamma_b[:, :K], online_coder_full.rho_b[:, :K])

# <codecell>

Et = tagger.transform(X_test)

# <codecell>

tags_predicted = Et.dot(online_coder_full.Eb[:, K:])
n_samples, n_tags = tags_predicted.shape

print tags_predicted.min(), tags_predicted.max()

div_factor = 3
tags_predicted = tags_predicted - div_factor * np.mean(tags_predicted, axis=0)

# <codecell>

predictat = 20
tags_predicted_binary = construct_pred_mask(tags_predicted, predictat)
tags_true_binary = (y_test > 0)

# <codecell>

prec, recall = per_tag_prec_recall(tags_predicted_binary, tags_true_binary)
print np.mean(prec), np.std(prec) / sqrt(n_tags)
print np.mean(recall), np.std(recall) / sqrt(n_tags)

# <codecell>

import cPickle as pickle

# <codecell>

with open('online_NMF_K%d_N%d_S%d.cPickle' % (K, n_components, online_coder_full.batch_size), 'wb') as f:
    pickle.dump(online_coder_full, f)

# <codecell>


