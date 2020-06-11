from sklearn.cluster import KMeans

import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense
from keras.optimizers import RMSprop
import matplotlib
matplotlib.use('PS')
matplotlib.use(
    "TkAgg"
)  # suggested to avoid AttributeError: 'FigureCanvasMac' object has no attribute 'renderer'
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import RandomUniform, Initializer, Constant
from keras.losses import binary_crossentropy

class InitCentersKMeans(Initializer):
    """ Initializer for initialization of centers of RBF network
        by clustering the given data set.
    # Arguments
        X: matrix, dataset
    """

    def __init__(self, X, max_iter=100):
        self.X = X
        self.max_iter = max_iter

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]

        n_centers = shape[0]
        km = KMeans(n_clusters=n_centers, max_iter=self.max_iter, verbose=0)
        km.fit(self.X)
        #PLOTTING THE KMEANS INIT CLUSTERS
        #print(km.cluster_centers_.shape)
        #plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1])
        #plt.show()
        return km.cluster_centers_



class InitCentersRandom(Initializer):
    """ Initializer for initialization of centers of RBF network
        as random samples from the given data set.
    # Arguments
        X: matrix, dataset to choose the centers from (random rows
          are taken as centers)
    """

    def __init__(self, X):
        self.X = X

    def __call__(self, shape, dtype=None):
        print("shapes", shape[1], self.X.shape[1])
        assert shape[1] == self.X.shape[1]
        idx = np.random.randint(self.X.shape[0], size=shape[0])
        return self.X[idx, :]


class RBFLayer(Layer):
    """ Layer of Gaussian RBF units.
    # Example
    ```python
        model = Sequential()
        model.add(RBFLayer(10,
                           initializer=InitCentersRandom(X),
                           betas=1.0,
                           input_shape=(1,)))
        model.add(Dense(1))
    ```
    # Arguments
        output_dim: number of hidden units (i.e. number of outputs of the
                    layer)
        initializer: instance of initiliazer to initialize centers
        betas: float, initial value for betas
    """

    def __init__(self, output_dim, initializer=None, betas=1.0, **kwargs):
        self.output_dim = output_dim
        self.init_betas = betas
        if not initializer:
            self.initializer = RandomUniform(0.0, 1.0)
        else:
            self.initializer = initializer
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=True)
        self.betas = self.add_weight(name='betas',
                                     shape=(self.output_dim,),
                                     initializer=Constant(
                                         value=self.init_betas),
                                     # initializer='ones',
                                     trainable=True)

        super(RBFLayer, self).build(input_shape)

    def call(self, x):

        C = K.expand_dims(self.centers)
        H = K.transpose(C-K.transpose(x))
        ret =  K.exp(-self.betas * K.sum(H**2, axis=1))
        return ret 

         #for diff
         #C = self.centers[np.newaxis, :, :]
         #X = x[:, np.newaxis, :]

         #diffnorm = K.sum((C-X)**2, axis=-1)
         #ret = K.exp( - self.betas * diffnorm)
         #return ret

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


  


def load_data():

    import sklearn.datasets
    n_samples = 1000
    #z, _  = sklearn.datasets.make_swiss_roll(n_samples=n_samples, noise=noise, random_state=None)
    X, y = sklearn.datasets.make_moons(n_samples=n_samples, noise=.02)
    # Anisotropicly distributed data
    #random_state = 170
    #X, y = sklearn.datasets.make_blobs(n_samples=n_samples, random_state=random_state, centers=3)
    #transformation = [[0.6, -0.6], [-0.4, 0.8]]
    #X = np.dot(X, transformation)
    return X, y


if __name__ == "__main__":

    X, y = load_data()
    #plt.scatter(X[:,0], X[:,1])
    #plt.show()
    inputlayer = keras.Input((2,)) # input

    rbflayer = RBFLayer(64,
                        initializer=InitCentersKMeans(X),#InitCentersRandom(X),
                        betas=5.0, # determine the sharpness of the gausian https://towardsdatascience.com/most-effective-way-to-implement-radial-basis-function-neural-network-for-classification-problem-33c467803319
                        input_shape=(2,))

    dense = Dense(1, activation='sigmoid', name='foo')
    rbf_only = rbflayer(inputlayer)
    rbf = dense(rbflayer(inputlayer))
    #model.add(Dense(1, name='foo'))
    model = Model(inputlayer, rbf)
    model.compile(loss=binary_crossentropy,
                  optimizer=RMSprop()) #'mean_squared_error',

    print(np.unique(y))# WE ONLY WANT 1 CLASSS SO WE CAN GET PROBABILITY COMPARED TO "BACKGROUNS"!!! SOONLY ADD ONES
    model.fit(X, np.ones(y.shape),
              batch_size=50,
              epochs=10,
              verbose=1)

    extractor = Model(inputlayer, rbf_only)
    feature = extractor.predict(X)

    print("features", feature.shape, feature[0], np.count_nonzero(feature[0]))

    y_pred = model.predict(X)
    print(y_pred[:10], y_pred.shape)
    #print(rbflayer.get_weights())

    
 

    centers = rbflayer.get_weights()[0]
    widths = rbflayer.get_weights()[1]
    print(centers.shape,widths.shape)
    print(centers[:,1].shape, centers[:,1].shape, np.zeros(len(centers)).shape)

    #plt.scatter(centers, np.zeros(len(centers)), s=20*widths)
    print(widths)
    
    print(centers[:10])

    plt.scatter(X[:,0], X[:,1], c=y_pred)
    #plt.scatter(centers[:,0], centers[:,1], s=20*widths, color="red")
    plt.colorbar()
    plt.show()


    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    end = max(x_max, y_max)
    xx, yy = np.meshgrid(np.arange(x_min, end, h),
                        np.arange(y_min, x_max, h))
    print(xx.shape, yy.shape)
    grid_inp = np.c_[xx.ravel(), yy.ravel()]
    
    Z = np.absolute(model.predict(grid_inp))
    print("MAXMIN", Z.min(), Z.max())

    def normalizeData(data):
      return (data - np.min(data)) / (np.max(data) - np.min(data))
    #Z = normalizeData(Z) + 0.001
    #print("MAXMIN", Z.min(), Z.max())
    # Put the result into a color plot
    Z =Z.reshape(xx.shape)
    print("MAXMIN", Z.min(), Z.max())
    print(Z.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z)# norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()),)
    # Plot also the training points
    #plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")

    plt.colorbar()
    plt.show()
    
    import sys, os
    sys.path.insert(0, os.path.abspath('../..'))
    from scripts.riemannian.riemannian_latent_space import RiemannianMetric, RiemannianTree
    import tensorflow as tf

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    
    rmetric = RiemannianMetric(x=model.get_layer(index = 1).output, z=model.input, session=session)
    #rmetric = RiemannianMetric(x=model.output, z=model.input, session=session)
    rmetric.create_tf_graph()

    mf = session.run(rmetric.MF, {rmetric.z: grid_inp})
    mf = np.nan_to_num(mf)
    print("MAXMIN", mf.min(), mf.max())

    Z = mf.reshape(xx.shape)
    print("MAXMIN", Z.min(), Z.max())
    #Z = (1/mf).reshape(xx.shape)
    #print("MAXMIN", Z.min(), Z.max())
    plt.pcolormesh(xx, yy, Z)
    plt.show()
    
    #print("MAXMIN", Z.min(), Z.max())
    Z = np.clip(normalizeData(Z), 0.0001, 0.9999)
    plt.pcolormesh(xx, yy, Z)
    plt.show()
    print("MAXMIN", Z.min(), Z.max())
    Z= 1/Z
    print("MAXMIN", Z.min(), Z.max())
    #Z=  np.nan_to_num(z_inverse)
    #print("MAXMIN", z_inverse.min(), z_inverse.max())
    



    plt.figure()
    plt.pcolormesh(xx, yy, Z)#, norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()))

    #plot actuals
    #plt.scatter(X[:,0], X[:,1], c=y_pred)
    plt.title("Full mf showing how inp space will change ")
    plt.colorbar()


    plt.show()


