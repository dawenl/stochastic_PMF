Poisson matrix factorization and automatic music tagging
======

Source code for the paper:
[Codebook-based Scalable Music Tagging with Poisson Matrix Factorization](http://dawenl.github.io/publications/LiangPE14-codebook.pdf) by Dawen Liang, John Paisley and Dan Ellis, in ISMIR 2014. 

Dawen Liang
dliang@ee.columbia.edu

(C) Copyright 2014, Dawen Liang

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.


### What's included:

There are four ipython notebook files, which will help reproduce the experiments in the aforementioned paper:

* [**buildVQ_MSD.ipynb**](http://nbviewer.ipython.org/github/dawenl/stochastic_PMF/blob/master/code/buildVQ_MSD.ipynb): Build the VQ Codebook for the Million Song Dataset and vector-quantize the MSD and save to disk.

* [**processLastfmTags.ipynb**](http://nbviewer.ipython.org/github/dawenl/stochastic_PMF/blob/master/code/processLastfmTags.ipynb): Process the tagging data from Last.fm and build the vocabulary and bag-of-tags representation and save to disk.

* [**tagging_ooc.ipynb**](http://nbviewer.ipython.org/github/dawenl/stochastic_PMF/blob/master/code/tagging_ooc.ipynb): After building the VQ-histogram and bag-of-tags, this one will reproduce the results from the data saved on the disk.

* [**tagging_in_memory.ipynb**](http://nbviewer.ipython.org/github/dawenl/stochastic_PMF/blob/master/code/tagging_in_memory.ipynb): If you have enough memory, you can also save the data from **tagging_ooc.ipynb** and directly fit to the PMF with this notebook.

### Dependencies:
* numpy 
* scipy
* scikit-learn (for evaluation metrics)
* nltk (for tag stemming)
