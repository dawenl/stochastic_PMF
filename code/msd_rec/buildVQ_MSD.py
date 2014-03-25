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
MSD_LFM_ROOT = os.path.join(MSD_DIR, 'Lastfm')
MSD_ADD = os.path.join(MSD_DIR, 'AdditionalFiles')

# <headingcell level=1>

# Building Codebook from a combination of "hot" and not-"hot" songs

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

def build_codewords(msd_data_root, tracks, cluster=None, n_clusters=2, max_iter=10, random_state=None):
    if type(random_state) is int:
        np.random.seed(random_state)
    elif random_state is not None:
        np.random.setstate(random_state)
        
    if cluster is None:    
        cluster = HartiganOnline.HartiganOnline(n_clusters=n_clusters)
    
    for i in xrange(max_iter):
        print 'Iteration %d: passing through the data...' % (i+1)
        for d in data_generator(msd_data_root, tracks):
            cluster.partial_fit(d)
    return cluster

# <codecell>

K = 512
cluster = build_codewords(MSD_DATA_ROOT, tracks_VQ, n_clusters=K, max_iter=3, random_state=98765)

# <codecell>

figure(figsize=(22, 4))
imshow(cluster.cluster_centers_.T, cmap=cm.PuOr_r, aspect='auto', interpolation='nearest')
colorbar()

# <codecell>

with open('Codebook_K%d_Hartigan' % K, 'wb') as f:
    pickle.dump(cluster, f)

# <headingcell level=1>

# Load last.fm tagging data

# <codecell>

ls "$MSD_LFM_ROOT"

# <codecell>

# get all the unique tags and the corresponding counts
uniq_tag_f = os.path.join(MSD_LFM_ROOT, 'unique_tags.txt')

# <codecell>

filtered_tags = (# favorate/like/love/blabla
                 'favorites', 'Favorite', 'Favourites', 'favourite', 'favorite songs', 'Favourite Songs', 'favorite song', 
                 'songs i love', 'lovedbybeyondwithin', 'Love it', 'love at first listen', 'fav', 'my favorite', 'top 40', 
                 'songs I absolutely love', 'favs', 'My Favorites', 'Favorite Artists', 'All time favourites', 'personal favourites',
                 'favouritestreamable', 'favorite tracks', 'Favorite Bands', 'like it', 'I love this song', 'rex ferric faves',
                 'love to death', 'my gang 09', 'My Favourites', 'BeatbabeBop selection', 'I Like It', 'newbest', 'top',
                 'IIIIIIIIII AMAZING TRACK :D IIIIIIIIII', 'best songs of the 80s', 'LOVE LOVE LOVE', 'i love it', 'most loved',
                 'favorite by this group', 'amayzes loved', 'DJPMan-loved-tracks', 'best of 2008', 'loved', 'Makes Me Smile',
                 '77davez-all-tracks', 'My pop music', 'best songs ever', 'favorite by this singer', 'I like', 'my music',
                 'Soundtrack Of My Life', 'UK top 40', 'Like', 'malloy2000 playlist - top songs - classical to metal', 'loved tracks',
                 'top artists', 'all time favorites', 'best songs of the 00s', 'favourite tracks', 'Solomusika-Loved', 
                 'all time faves', 'british i like', 'Jills Station', 'de todo mio favoritos', 'Faves', 'Fave',

                 # great/awesome/blabla
                 'kick ass', 'wonderful', 'excellent', 'Great Lyricists', 'badass', 'awesomeness', 'great song', 'Awesome', 'cool',  
                 'amazing', 'good', 'nice', 'sweet', 'best', 'FUCKING AWESOME', 'lovely', 'Good Stuff', 'brilliant', 'feel good', 
                 'perfect', 'all the best', 'cute', 'the best', '<3', 'interesting', 'feelgood', 'pretty', 'i feel good', 'good shit',
                 'good music', 'good song', 'great songs', 'yeah', 'best song ever', 'wow', 'worship', 'makes me happy', 'ok',
                 'damned good', 'underrated', 'Perfection',
                 
                 # rating
                 '1', '3', '4', '5', '4 Stars', '3 stars', '4 Star', '3 star', '3-star',
                 
                 # year
                 '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', 
                 '2006', '2007', '2008', '2009', '2010',
                 
                 # descriptive
                 'songwriter', 'singer-songwriter', 'cover', 'covers', 'seen live', 'heard on Pandora', 'title is a full sentence',
                 'Retro', 'Miscellaneous', 'collection', 'billboard number ones', 'ost', 'cover song', 'singer songwriter', 'new',
                 'download', 'over 5 minutes long', 'Soundtracks', 'under two minutes', 'albums I own', 'cover songs', 'Radio',
                 'heard on last-fm',
      
                 # I don't know what you are talking about
                 'buy', 'lol', 'us', 'other', '2giveme5', 'i am a party girl here is my soundtrack', 'names', 'Tag', 'check out',
                 'f', 'test', 'out of our heads', 'me', 'I want back to the 80s', '9 lbs hammer', 'yes',
                 )  

# <codecell>

tags = set()

# we only pick the tags with >= 1000 counts, otherwise it's just too noisy
# e.g. "writing papers to pay for the college you have gotten into" has 13 counts 
with open(uniq_tag_f, 'rb') as f:
    for line in f:
        try:
            tag, count = line.strip().split('\t', 2)
            if int(count) >= 1000:
                if not tag in filtered_tags:
                    tags.add(sanitize(tag))
                pass
            else:
                # since the file is ordered by count
                break
        except ValueError as e:
            print 'The following line raises the error:', e
            # there is one line with no tag information, but with less than 1000 counts
            print line

# <codecell>

len(tags)

# <codecell>

voc = sorted(tags)

# <codecell>

def sanitize(tag):
    return tag.lower().replace("-", " ").replace("'", "''")

# <codecell>

voc

# <codecell>

redundant_map = {
                 '1950s': '50s', '1960s': '60s', '1970s': '70s', '1980s': '80s', '1990s': '90s', '2000s': '00s',
                 '50''s': '50s', '60''s': '60s', '70''s': '70s', '80''s': '80s', '90''s': '90s', 
                 'african': 'africa', 
                 
                 }

# <codecell>

# pickout corresponding data from training/test split
import sqlite3

# <codecell>

md_dbfile = os.path.join(MSD_ADD, 'track_metadata.db')

# <codecell>

conn = sqlite3.connect(md_dbfile)
c = conn.cursor()

# <codecell>

with open('tracks_tag_train.txt', 'wb') as fw:
    with open('artists_train.txt', 'rb') as fr:
        for line in fr:
            aid = line.strip()
            assert len(aid) == 18 and aid[:2] == 'AR'
            q = "SELECT track_id FROM songs WHERE artist_id='%s'" % aid
            res = c.execute(q)
            for r in res:
                tid = r[0]
                track_dir = os.path.join(MSD_LFM_ROOT, 'lastfm_train', '/'.join(tid[2:5]), tid + '.json')
                if os.path.exists(track_dir):
                    fw.write(tid + '\n')

# <codecell>

with open('tracks_tag_test.txt', 'wb') as fw:
    with open('artists_test.txt', 'rb') as fr:
        for line in fr:
            aid = line.strip()
            assert len(aid) == 18 and aid[:2] == 'AR'
            q = "SELECT track_id FROM songs WHERE artist_id='%s'" % aid
            res = c.execute(q)
            for r in res:
                tid = r[0]
                track_dir = os.path.join(MSD_LFM_ROOT, 'lastfm_test', '/'.join(tid[2:5]), tid + '.json')
                if os.path.exists(track_dir):
                    fw.write(tid + '\n')

# <codecell>


