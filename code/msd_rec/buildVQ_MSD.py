# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import glob
import os

import hdf5_getters
import HartiganOnline, VectorQuantizer

# <codecell>

MSD_DIR = u'/q/boar/boar-p9/MillionSong/'

# <codecell>

cd /q/boar/boar-p9/MillionSong/AdditionalFiles/

# <codecell>

!grep "Jordan Rudess" unique_tracks.txt

# <codecell>

MSD_DATA_ROOT = os.path.join(MSD_DIR, 'data')

def get_all_song_hotttnesss(basedir, ext='.h5') :
    track_to_hotttnesss = dict()
    with open(os.path.join(MSD_DIR, 'AdditionalFiles', 'unique_tracks.txt'), 'rb') as f:
        for (count, line) in enumerate(f):
            track_ID, _, _, _ = line.strip().split('<SEP>')
            track_dir = os.path.join(MSD_DATA_ROOT, '/'.join(track_ID[2:5]), track_ID + ext)
            h5 = hdf5_getters.open_h5_file_read(track_dir)
            hotttnesss = hdf5_getters.get_song_hotttnesss(h5)
            if not math.isnan(hotttnesss):
                track_to_hotttnesss[track_ID] = hotttnesss
            h5.close()  
            if not count % 500:
                print "%7d tracks processed" % count 
    return track_to_hotttnesss

# <codecell>

track_to_hotttnesss = get_all_song_hotttnesss(MSD_DIR_BASE)

# <codecell>


