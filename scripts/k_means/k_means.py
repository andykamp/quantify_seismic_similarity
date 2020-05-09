from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import copy

def k_mean_algorithm(Xtrain, n_clusters=3, z_index=False):
    X= copy.deepcopy(Xtrain)
    print("Starting Kmeans: ", X.shape,)
    X[X == 0] = 12345 # set all zeros to a givenvalue to get coords of nonzero and then setting it back later
    x,y,z,_ = X.nonzero() # extract all cordinates that are not zero
    print("Total coords: ", X.shape[0]*X.shape[1]*X.shape[2]*X.shape[3])
    coord_amplitude = X[x,y,z] # one rank array of
    features = np.empty((x.shape[0], 4))
    print('coord_amplitude.shape', coord_amplitude.shape)
    print('x.shape', x.shape)
    print('y.shape', y.shape)
    print('z.shape', z.shape)
    print('features.shape', features.shape)

    numpy_feature = np.column_stack((x,y))
    numpy_feature = np.column_stack((numpy_feature, z))
    numpy_feature = np.column_stack((numpy_feature))
    numpy_feature[numpy_feature == 12345] = 0
    print("Kmean input shape", numpy_feature.shape )

    if type(n_clusters) is list:
        for class_ in n_clusters:
            clf = KMeans(n_clusters=n_clusters)
            if z_index:
                unsupervised_output = clf.fit_predict(np.column_stack((coord_amplitude, z)))
            else:
                unsupervised_output = clf.fit_predict(np.column_stack((coord_amplitude)))
            #TODO: PLOT
    else:
        clf = KMeans(n_clusters=n_clusters)
        if z_index:
            unsupervised_output = clf.fit_predict(np.column_stack((coord_amplitude, z)))
        else:
            unsupervised_output = clf.fit_predict(coord_amplitude)
        print("Kmean predict ouput shape", unsupervised_output.shape)
        unsupervised_output = unsupervised_output.reshape(X.shape)
        print("Kmean  ouput shape", unsupervised_output.shape)
    return unsupervised_output



def find_num_clusters(max_cluster,data):
    ks = range(1, max_cluster)
    inertias = []
    for k in ks:
        # Create a KMeans instance with k clusters: model
        model = KMeans(n_clusters=k)
        
        # Fit model to samples
        model.fit(data)
        
        # Append the inertia to the list of inertias
        inertias.append(model.inertia_)
        
    plt.plot(ks, inertias, '-o', color='black')
    plt.xlabel('number of clusters, k')
    plt.ylabel('inertia')
    plt.xticks(ks)
    plt.show()





def not_working(X,y=None):
    estimators = [('k_means_iris_8', KMeans(n_clusters=8)),
                  ('k_means_iris_3', KMeans(n_clusters=3)),
                  ('k_means_iris_bad_init', KMeans(n_clusters=3, n_init=1,
                                                   init='random'))]

    fignum = 1
    titles = ['8 clusters', '3 clusters', '3 clusters, bad initialization']
    for name, est in estimators:
        fig = plt.figure(fignum, figsize=(4, 3))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        est.fit(X)
        labels = est.labels_

        ax.scatter(X[:, 3], X[:, 0], X[:, 2],
                   c=labels.astype(np.float), edgecolor='k')

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('Petal width')
        ax.set_ylabel('Sepal length')
        ax.set_zlabel('Petal length')
        ax.set_title(titles[fignum - 1])
        ax.dist = 12
        fignum = fignum + 1

    # Plot the ground truth
    #fig = plt.figure(fignum, figsize=(4, 3))
    #ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    #for name, label in [('Setosa', 0),
    #                    ('Versicolour', 1),
    #                    ('Virginica', 2)]:
    #    ax.text3D(X[y == label, 3].mean(),
    #              X[y == label, 0].mean(),
    #              X[y == label, 2].mean() + 2, name,
    #              horizontalalignment='center',
    #              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    #y = np.choose(y, [1, 2, 0]).astype(np.float)
    #ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, edgecolor='k')

    #ax.w_xaxis.set_ticklabels([])
    #ax.w_yaxis.set_ticklabels([])
    #ax.w_zaxis.set_ticklabels([])
    #ax.set_xlabel('Petal width')
    #ax.set_ylabel('Sepal length')
    #ax.set_zlabel('Petal length')
    #ax.set_title('Ground Truth')
    #ax.dist = 12

    fig.show()


def plot_kmeans(Xtrain):
    ### Plot K_means output
    #inverse_k_means = pca.inverse_transform(unsupervised_pred)
    #inverse_reshaped_k_means = inverse_k_means.reshape(Xtrain.shape)
    
    # find the number of cluster 
    #find_num_clusters(10,PCA_components.iloc[:,:3])


    ### Plot xline before and after
    # plots original random xline
    plt.title("Original xline")
    plt.imshow(np.squeeze(Xtrain[0]), cmap = "gray")
    plt.colorbar()
    #plt.show()
    plt.savefig('1.input.jpg')


    # pnormal k-means using 3 clusters and only amplitude
    print("\nStarting k_means...")
    unsupervised_pred = k_mean_algorithm(Xtrain, n_clusters=3)
    print("Done with k_means\n")
    K_MEANS=np.squeeze(unsupervised_pred[0])
    plt.title("Kmeans prediction inline")
    plt.imshow(K_MEANS, cmap = "gist_rainbow")
    plt.colorbar()
    #plt.show()
    plt.savefig('2.kmeans_3_clusters.jpg')


    # how does it look if we multiple the orgiinal intensities with the now clusetered seimsic?
    K_MEANS_ampl = unsupervised_pred*Xtrain
    plt.title("Kmeans*amplitude prediction inline")
    plt.imshow(np.squeeze(K_MEANS_ampl[0,]), cmap = "gist_rainbow")
    plt.colorbar()
    #plt.show()
    plt.savefig('3.kmeans_3_clusters_mulitply_amplitude.jpg')



    ############# TAKES TIME TO COMPUTE #######################
    # k-means on k-meaned data with coordinates as a feature 
    print("\nStarting k_means..." )
    temp1 = k_mean_algorithm(copy.deepcopy(unsupervised_pred), n_clusters=50, z_index=True)
    print("Done with k_means\n")
    plt.title("Kmeans of kmeans prediction inline")
    plt.imshow(np.squeeze(temp1[0]), cmap = "gist_rainbow")
    plt.colorbar()
    plt.show()

    
    # k-means on k-meaned + amplitude data with coordinates as a feature 
    print("\nStarting k_means...")
    temp2 = k_mean_algorithm(K_MEANS_ampl, n_clusters=50, z_index=True)
    print("Done with k_means\n")
    plt.title("K_MEANS_ampl K_MEANS_ampl prediction inline")
    plt.imshow(np.squeeze(temp2[0]), cmap = "gist_rainbow")
    plt.colorbar()
    plt.show()

    ##################################################################


    # treshhold data
    data_ = np.squeeze(Xtrain[0])
    data = copy.deepcopy(data_)
    data[data<10] = 0
    plt.title("Kmeans*amplitude prediction inline")
    plt.imshow(data, cmap = "gist_rainbow")
    plt.colorbar()
    plt.show()

    data[data<20] = 0
    plt.title("Kmeans*amplitude prediction inline")
    plt.imshow(data, cmap = "gist_rainbow")
    plt.colorbar()
    plt.show()
    data[data<30] = 0
    plt.title("Kmeans*amplitude prediction inline")
    plt.imshow(data, cmap = "gist_rainbow")
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    #get data to use 
    data_cube = np.load("/Users/anderskampenes/Documents/Dokumenter/NTNU/MASTER/code/data/raw/f3-benchmark/test_once/test1_seismic.npy")
    # make sure it is fomrated correctly 
    img = data_cube[0:10,:,:].T
    img = np.transpose(img,(2,0,1))
    print(img.shape)
    data = img.reshape(img.shape[0], img.shape[1], img.shape[2],1)
    #plot and perform kmeans 
    plot_kmeans(data)



    