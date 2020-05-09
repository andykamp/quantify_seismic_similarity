# Author: Tom Dupré la Tour
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.datasets import make_blobs

print(__doc__)
#get data to use 
data_cube = np.load("/Users/anderskampenes/Documents/Dokumenter/NTNU/MASTER/code/data/raw/f3-benchmark/test_once/test1_seismic.npy")
# make sure it is fomrated correctly 
img = data_cube[:,0,:].T

n_bins = 3
strategies = ['uniform', 'quantile', 'kmeans']

# construct the datasets
X_list = [
    img
]

figure = plt.figure(figsize=(14, 9))
i = 1
for ds_cnt, X in enumerate(X_list):

    ax = plt.subplot(len(X_list), len(strategies) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data", size=14)

    plt.imshow(img)


    i += 1
    # transform the dataset with KBinsDiscretizer
    for strategy in strategies:
        enc = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
        enc.fit(X)
        grid_encoded = enc.transform(X)

        ax = plt.subplot(len(X_list), len(strategies) + 1, i)
        plt.imshow(grid_encoded)
       
        if ds_cnt == 0:
            ax.set_title("strategy='%s'" % (strategy, ), size=14)
        i += 1

plt.tight_layout()
#plt.show()
plt.savefig(f'KBinsDiscretizer_bins:{n_bins}')