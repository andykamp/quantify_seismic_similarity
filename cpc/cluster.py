import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import os
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

# Importing sklearn and TSNE.
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)
#from sklearn.utils.extmath import _ravel
# Random state we define this random state to use this value in TSNE which is a randmized algo.
RS = 25111993

# Importing matplotlib for graphics.
import matplotlib
#matplotlib.use(
#    "TkAgg"
#)  # suggested to avoid AttributeError: 'FigureCanvasMac' object has no attribute 'renderer'

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

# Importing seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})



""" 
Fetch from https://towardsdatascience.com/plotting-text-and-image-vectors-using-t-sne-d0e43e55d89
"""
def K_MEANS(data, n_clusters):# Here we are importing KMeans for clustering Product Vectors
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters, random_state=0).fit(data)
    # We can extract labels from k-cluster solution and store is to a list or a vector as per our requirement
    Y=kmeans.labels_ # a vector
    z = pd.DataFrame(Y.tolist()) # a list
    return Y, z

def T_SNE(data):
    # Fit the model using t-SNE randomized algorithm
    digits_proj = TSNE(random_state=RS).fit_transform(data)
    return digits_proj

# An user defined function to create scatter plot of vectors
def scatter(x, colors, n_clusters, color_map="hls", show_labels=False):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette(color_map, n_clusters))

    # We create a scatter plot.
    f = plt.figure(figsize=(32, 32))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=120,
                    c=palette[colors.astype(np.int)])
    #plt.xlim(-25, 25)
    #plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each cluster.
    txts = []
    if show_labels:
        for i in range(n_clusters):
            # Position of each label.
            xtext, ytext = np.median(x[colors == i, :], axis=0)
            txt = ax.text(xtext, ytext, str(i), fontsize=20)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)

    return f, ax, sc, txts


if __name__ == "__main__":
    # load data 
    data = np.random.rand(100,128)
    # define the number fo clusters 
    n_clusters = 18
    # generate clusters 
    Y, z=K_MEANS(data, n_clusters)
    # generate t-sne projections
    digits_proj = T_SNE(data)
    print(list(range(0,n_clusters)))
    sns.palplot(np.array(sns.color_palette("hls", n_clusters)))
    scatter(digits_proj, Y, n_clusters)
    plt.savefig('digits_tsne-generated_18_cluster.png', dpi=120)