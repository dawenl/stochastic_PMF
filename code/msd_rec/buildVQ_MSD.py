# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import cPickle as pickle
import itertools
import json
import operator
import os

import hdf5_getters
import HartiganOnline, VectorQuantizer

# <codecell>

MSD_DIR = u'/q/boar/boar-p9/MillionSong/'
MSD_DATA_ROOT = os.path.join(MSD_DIR, 'data')
MSD_ADD = os.path.join(MSD_DIR, 'AdditionalFiles')

# <codecell>

# get all the tracks with non-nan hotttnesss
def get_all_song_hotttnesss(msd_dir, ext='.h5') :
    track_to_hotttnesss = dict()
    msd_data_root = os.path.join(msd_dir, 'data')
    with open(os.path.join(msd_dir, 'AdditionalFiles', 'unique_tracks.txt'), 'rb') as f:
        for (count, line) in enumerate(f):
            track_ID, _, _, _ = line.strip().split('<SEP>')
            track_dir = os.path.join(msd_data_root, '/'.join(track_ID[2:5]), track_ID + ext)
            h5 = hdf5_getters.open_h5_file_read(track_dir)
            hotttnesss = hdf5_getters.get_song_hotttnesss(h5)
            if not math.isnan(hotttnesss):
                track_to_hotttnesss[track_ID] = hotttnesss
            h5.close()  
            if not count % 1000:
                print "%7d tracks processed" % count 
    return track_to_hotttnesss

# <codecell>

if os.path.exists('track_to_hotttnesss.json'):
    with open('track_to_hotttnesss.json', 'rb') as f:
        track_to_hotttnesss = json.load(f)
else:
    track_to_hotttnesss = get_all_song_hotttnesss(MSD_DIR)
    with open('track_to_hotttnesss.json', 'wb') as f:
        json.dump(track_to_hotttnesss, f)

# <codecell>

# see some track-hotttnesss pairs
track_to_hotttnesss_ordered = sorted(track_to_hotttnesss.iteritems(), key=operator.itemgetter(1), reverse=True)
for i in xrange(0, 50000, 1000):
    track_ID = track_to_hotttnesss_ordered[i][0]
    hotttnesss = track_to_hotttnesss_ordered[i][1]
    out = !grep "$track_ID" "$MSD_ADD"/unique_tracks.txt 
    print out[0].strip().split('<SEP>')[2:4], 'Hotttnesss:', hotttnesss

# <codecell>

# and see how the hotttnesss are distributed
hist(track_to_hotttnesss.values(), bins=20)
pass

# <codecell>

# randomly select 47500 non-zero-hotttnesss tracks and 2500 zeros-hotttnesss tracks
np.random.seed(98765)
tracks_nzhotttnesss = np.random.choice(filter(lambda x: track_to_hotttnesss[x] != 0.0, track_to_hotttnesss.keys()), 
                                       size=47500, replace=False)
tracks_zhotttnesss = np.random.choice(filter(lambda x: track_to_hotttnesss[x] == 0.0, track_to_hotttnesss.keys()), 
                                      size=2500, replace=False)
tracks_VQ = np.hstack((tracks_nzhotttnesss, tracks_zhotttnesss))

# <codecell>

def data_generator(msd_data_root, tracks, shuffle=True, ext='.h5'):
    if shuffle:
        np.random.shuffle(tracks)
    for track_ID in tracks:
        track_dir = os.path.join(msd_data_root, '/'.join(track_ID[2:5]), track_ID + ext)
        h5 = hdf5_getters.open_h5_file_read(track_dir)
        mfcc = hdf5_getters.get_segments_timbre(h5)
        h5.close()
        if shuffle:
            np.random.shuffle(mfcc)
        yield mfcc

# <codecell>

def build_codewords(msd_data_root, tracks, K, max_iter=10, random_state=None):
    if type(random_state) is int:
        np.random.seed(random_state)
    else:
        np.random.setstate(random_state)
        
    cluster = HartiganOnline.HartiganOnline(n_clusters=K)
    for i in xrange(max_iter):
        print 'Iteration %d: passing through the data...' % (i+1)
        for d in data_generator(msd_data_root, tracks):
            cluster.partial_fit(d)
    return cluster

# <codecell>

K = 512
cluster = build_codewords(MSD_DATA_ROOT, tracks_VQ, K, max_iter=3, random_state=98765)

# <codecell>

figure(figsize=(22, 4))
imshow(cluster.cluster_centers_.T, cmap=cm.PuOr_r, aspect='auto', interpolation='nearest')
colorbar()

# <codecell>

figure(figsize=(22, 4))
imshow(cluster.cluster_centers_.T, cmap=cm.PuOr_r, aspect='auto', interpolation='nearest')
colorbar()

# <codecell>

with open('Codebook_K%d_Hartigan' % K, 'wb') as f:
    pickle.dump(cluster, f)

