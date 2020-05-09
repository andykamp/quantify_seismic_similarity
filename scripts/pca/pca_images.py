from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from sklearn.decomposition import PCA

def pca_sklearn(Xtrain, Ytrain=None, plot_scatter=False, floydhub=False):
    # pca expects shape of (num_samples, num_features)
    print("Pre-PCA shape:", Xtrain.shape)
    nsamples, nx, ny, channels = Xtrain.shape

    pca_train_dataset = Xtrain.reshape((nsamples, nx*ny*channels))
    print("pca_train_dataset", pca_train_dataset.shape)
    


    ## Performing PCA
    num_red_dims=0.95 # TODO, make dependent to the  x % acccuracy needed

    print("\nStarting PCA on dataset...")
    pca = PCA(num_red_dims)
    print("pca inited")
    reduced = pca.fit_transform(pca_train_dataset)
    print("Done with PCA on dataset!\n")
    if plot_scatter:
        plt.scatter(reduced[:,0], reduced[:,1], s=100, c=Ytrain, alpha=0.5)
        plt.show()
    # When talking about PCA, the sum of the sample variances of all individual
    # variables is called the total variance. If we divide individual variances
    # by the total variance, we’ll see how much variance each variable explainsself.
    # This is plotted below, and should be held by the leftmost part of the graph,
    # seeing it is sorted in decreased order with regards to variance.
    plt.plot(pca.explained_variance_ratio_)
    if floydhub:
        plt.savefig('/output/PCA_explained_variance_ratio' , bbox_inches='tight')
        plt.clf()
    else:
        plt.show()
    # cumulative variance, vizualising how many principal components is needed to obtain 100% of the variance
    # choose k = number of dimensions that gives us 95-99% variance
    cumulative = []
    last = 0
    for v in pca.explained_variance_ratio_:
        cumulative.append(last + v)
        last = cumulative[-1]
    plt.plot(cumulative)
    if floydhub:
        plt.savefig('/output/PCA_cumulative' , bbox_inches='tight')
        plt.clf()
    else:
        plt.show()


   

    # Plot the explained variances
    features = range(pca.n_components_)
    plt.bar(features, pca.explained_variance_ratio_, color='black')
    plt.xlabel('PCA features')
    plt.ylabel('variance %')
    plt.xticks(features)
    plt.show()

     # Save components to a DataFrame
    PCA_components = pd.DataFrame(reduced)
    ## we plot the first two dimensjons. We do this to notice if there is any clear clusters
    plt.scatter(PCA_components[0], PCA_components[1], alpha=.1, color='black')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.show()


    ### calculate PCA and inverse transform PCA output
    reduced_img = pca.inverse_transform(reduced)
    reduced_reshaped_img = reduced_img.reshape(Xtrain.shape)
    print("reduced.shape", reduced.shape)
    print("reduced_img.shape", reduced_img.shape)
    print("reduced_img.reshape(Xtrain.shape)", reduced_img.reshape(Xtrain.shape).shape)

    plot_PCA=True
    if plot_PCA:

        ### PLot inline before and after
        # plots original random iline
        plt.title("Original image")
        plt.imshow(np.squeeze(Xtrain[0]))
        plt.colorbar()
        plt.show()
        plt.show()
        # plots PCA reduced random inline
        plt.title("PCA transformed image")
        plt.imshow(np.squeeze(reduced_reshaped_img[0]))
        plt.colorbar()

        if floydhub:
            plt.savefig('/output/PCA_reduced_img' , bbox_inches='tight')
            plt.clf()
        else:
            plt.show()

    return reduced_reshaped_img



if __name__ == '__main__':
    #get data to use 
    data_cube = np.load("/Users/anderskampenes/Documents/Dokumenter/NTNU/MASTER/code/data/raw/f3-benchmark/test_once/test1_seismic.npy")
    # make sure it is fomrated correctly 
    img = data_cube[0:10,:,:].T
    img = np.transpose(img,(2,0,1))
    print(img.shape)
    data = img.reshape(img.shape[0], img.shape[1], img.shape[2],1)
    pca_sklearn(data)
