# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import cPickle as pickle
import os
import re
import sqlite3

import nltk
from nltk.stem.porter import PorterStemmer

# <codecell>

MSD_DIR = u'/q/boar/boar-p9/MillionSong/'
MSD_LFM_ROOT = os.path.join(MSD_DIR, 'Lastfm')
MSD_ADD = os.path.join(MSD_DIR, 'AdditionalFiles')

# <codecell>

tags_dbfile = os.path.join(MSD_LFM_ROOT, 'lastfm_tags.db')
uniq_tag_f = os.path.join(MSD_LFM_ROOT, 'unique_tags.txt')
md_dbfile = 'track_metadata.db'

# <codecell>

# shameless steal from https://github.com/bmcfee/hypergraph_playlist/blob/master/buildTagmatrix.py
def getVocab(dbc):
    vocab = []
    cur = dbc.cursor()
    cur.execute('''SELECT tag FROM tags''')
    for (term,) in cur:
        vocab.append(term)
        pass
    return vocab

def getTrackRows(dbc):
    cur = dbc.cursor()
    tid = {}
    cur.execute('''SELECT tid FROM tids''')
    for (i, (track,)) in enumerate(cur, 1):
        tid[track] = i
        pass
    return tid

# <codecell>

with sqlite3.connect(tags_dbfile) as dbc:
    vocab = getVocab(dbc)
    tid = getTrackRows(dbc)

# <codecell>

def tid_to_dir(base_dir, tid, ext='.h5'):
    return os.path.join(base_dir, '/'.join(tid[2:5]), tid + ext)

def sanitize(tag):
    return re.sub(r'(\W|_)+', '', re.sub('(&| n )', 'and', ' '.join([stemmer.stem(token) for token in nltk.word_tokenize(tag.lower())])))

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
                 'acclaimed music top 3000'

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
                 '2006', '2007', '2008', '2009', '2010', '00s', '10s', '1950s', '1960s', '1970s', '1980s', '1990s', '2000s', 
                 '20th Century', '21st century', "50's", '50s', "60's", '60s', '60s Gold', "70's", '70s', "80's", '80s', '80s Pop',
                 '80s rock', "90's", '90s', '90s Rock',
                 
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

stag_to_tag = dict()
stemmer = PorterStemmer()

# we only pick the tags with >= 1000 counts, otherwise it's just too noisy
# e.g. "writing papers to pay for the college you have gotten into" has 13 counts 
with open(uniq_tag_f, 'rb') as f:
    for line in f:
        try:
            tag, count = line.strip().split('\t', 2)
            if int(count) >= 1000:
                if not tag in filtered_tags:
                    stag = sanitize(tag)
                    if stag in stag_to_tag:
                        stag_to_tag[stag].append(tag)
                    else:
                        stag_to_tag[stag] = [tag]
            else:
                # since the file is ordered by count
                break
        except ValueError as e:
            print 'The following line raises the error:', e
            # there is one line with no tag information, but with less than 1000 counts
            print line

# <codecell>

tags = ['NO_TAGS'] + sorted(stag_to_tag.keys())

# <codecell>

voc_to_num = dict((tag, i) for (i, tag) in enumerate(tags))

# <codecell>

def getArtistTracks(cur, aid):
    cur.execute("SELECT track_id FROM songs WHERE artist_id='%s'" % aid)
    for (track, ) in cur_md:
        yield track
        
        
def getValidTrackTags(cur, track, tid, vocab, voc_to_num):
    cur_td.execute("SELECT tag, val FROM tid_tag WHERE tid = %d AND val > 0" % tid[track])
    out = {}
    for (tag, val) in cur_td:
        stag = sanitize(vocab[tag])
        if stag not in voc_to_num:
            continue
        if voc_to_num[stag] in out:          
            out[voc_to_num[stag]] += float(val)
        else:
            out[voc_to_num[stag]] = float(val)
    if len(out) == 0:
        out[voc_to_num['NO_TAGS']] = 1
    return out


def numberize(infile, outfile, cur_md, cur_td, tid, vocab, voc_to_num):
    with open(outfile, 'wb') as fw:
        with open(infile, 'rb') as fr:
            for line in fr:
                aid = line.strip()
                for track in getArtistTracks(cur_md, aid):
                    if track not in tid:
                        continue
                    out = getValidTrackTags(cur_td, track, tid, vocab, voc_to_num)
                    fw.write('%s\t%s\n' % (track, ' '.join('%d:%.1f' % pair for pair in out.items())))    

# <codecell>

# turn the whole MSD tags to numbers
conn_md = sqlite3.connect(md_dbfile)
conn_td = sqlite3.connect(tags_dbfile)

cur_md = conn_md.cursor()
cur_td = conn_td.cursor()

numberize('artists_train.txt', 'tracks_tag_train.num', cur_md, cur_td, tid, vocab, voc_to_num)
numberize('artists_test.txt', 'tracks_tag_test.num', cur_md, cur_td, tid, vocab, voc_to_num)

conn_md.close()
conn_td.close()

# <codecell>


