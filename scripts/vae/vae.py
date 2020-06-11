from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import metrics
from keras import backend as K
import matplotlib
matplotlib.use('PS')
matplotlib.use(
    "TkAgg"
)  # suggested to avoid AttributeError: 'FigureCanvasMac' object has no attribute 'renderer'
import matplotlib.pyplot as plt
import math 
from keras.datasets import mnist
import numpy as np
import tensorflow as tf

import sys, os
sys.path.insert(0, os.path.abspath('../..'))
from scripts.riemannian.riemannian_latent_space import RiemannianMetric, RiemannianTree



def buld_vae(original_dim, intermediate_dim, latent_dim, epochs):
    epsilon_std=1.
    x = Input(batch_shape=(batch_size, original_dim))
    h = Dense(intermediate_dim, activation='relu')(x)
    h2 = Dense(intermediate_dim, activation='relu')(h)
    z_mean = Dense(latent_dim)(h2)
    z_log_sigma = Dense(latent_dim)(h)


    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_sigma) * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_h2 = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    
    x_decoded_mean = decoder_mean(h_decoded)


    # end-to-end autoencoder
    vae = Model(x, x_decoded_mean)

    # encoder, from inputs to latent space
    encoder = Model(x, z_mean)

    # generator, from latent space to reconstructed inputs
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _h2_decoded = decoder_h2(_h_decoded)
    _x_decoded_mean = decoder_mean(_h2_decoded)
    generator = Model(decoder_input, _x_decoded_mean)


    def vae_loss(x, x_decoded_mean):
        xent_loss = metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = -5e-4 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
        return xent_loss + kl_loss

    vae.compile(optimizer='rmsprop', loss=vae_loss)
    return vae, encoder, generator

if __name__ == "__main__":
    
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    #image_size = x_train.shape[1]
    #original_dim = image_size * image_size
    #x_train = np.reshape(x_train, [-1, original_dim])
    #x_test = np.reshape(x_test, [-1, original_dim])
    #x_train = x_train.astype('float32') / 255
    #x_test = x_test.astype('float32') / 255


    import sklearn.datasets
    n_samples = 100000
    #z, _  = sklearn.datasets.make_swiss_roll(n_samples=n_samples, noise=noise, random_state=None)
    z, y = sklearn.datasets.make_moons(n_samples=n_samples, noise=.09)
    test_split = 0.1
    split = int(z.shape[0]*0.1)
    print("split", split)
    x_train, y_train = z[:split], y[:split]
    x_test, y_test = z[split:], y[split:]
    

    original_dim = 2
    intermediate_dim = 512
    latent_dim = 2
    batch_size= 100
    epsilon_std= 1.

    epochs = 1

    vae, encoder, generator = buld_vae(original_dim, intermediate_dim, latent_dim, epochs)
    vae.fit(x_train, x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, x_test))

    #vae.save_weights('vae_mlp_mnist.h5')

    #encode data 
    x_test_encoded = encoder.predict(x_train, batch_size=batch_size)
    # plot encoded test data with labels
    plt.figure(figsize=(6, 6))
    #plt.scatter(x_test[:,0], x_test[:,1], color="green")
    #plot encoded data
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_train)
    plt.colorbar()
    plt.title("Test data (green), Encoded (else)")
    plt.show()

    # plot reconstruced image
    #recon = generator.predict(x_test_encoded, batch_size=batch_size) 
    #print(recon.shape)
    #plt.imshow(recon[0].reshape(28,28))
    #plt.show()

    # display a 2D manifold of the digits
    #n = 15  # figure with 15x15 digits
    #digit_size = 28
    #figure = np.zeros((digit_size * n, digit_size * n))
    # we will sample n points within [-15, 15] standard deviations
    #grid_x = np.linspace(-15, 15, n)
    #grid_y = np.linspace(-15, 15, n)

    #for i, yi in enumerate(grid_x):
    #    for j, xi in enumerate(grid_y):
    #        z_sample = np.array([[xi, yi]]) * epsilon_std
    #        x_decoded = generator.predict(z_sample)
    #        digit = x_decoded[0].reshape(digit_size, digit_size)
    #        figure[i * digit_size: (i + 1) * digit_size,
    #            j * digit_size: (j + 1) * digit_size] = digit

    #plt.figure(figsize=(10, 10))
    #plt.imshow(figure)
    #plt.show()



    # calc riemann on encoder
    generator.summary()
    x_test_decoded = generator.predict(x_test_encoded, batch_size=batch_size)
    plt.scatter(x_test[:,0], x_test[:,1], color="green")

    plt.scatter(x_test_decoded[:batch_size,0], x_test_decoded[:batch_size,1], c="red")
    plt.title("Test data (green), Decoded (red)")
    plt.show()

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    rmetric = RiemannianMetric(x=generator.output, z=generator.input, session=session)
    rmetric.create_tf_graph()

    mf = session.run(rmetric.MF, {rmetric.z: x_test_encoded[:batch_size]})
    plt.figure()
    plt.scatter(x_test_encoded[:batch_size,0], x_test_encoded[:batch_size,1], c=mf)
    plt.title("Encoded with change magnitude color")
    plt.colorbar()
    plt.show()



    # plot riemann for each single gridpoint 
    # plot mf for entire grid 
    h = .09  # step size in the mesh
    x_min, x_max = x_test_encoded[:, 0].min() - 1, x_test_encoded[:, 0].max() + 1
    y_min, y_max = x_test_encoded[:, 1].min() - 1,x_test_encoded[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    grid_inp = np.c_[xx.ravel(), yy.ravel()] 
    grid_inp_shape = grid_inp.shape[0]

    # need to calc overflow as it needs to fit the batch size
    overflow = math.ceil(grid_inp_shape/batch_size)
    # need to batch up grid in batch sizes... 
    mf = np.empty([overflow*batch_size]) 
    for i in range(overflow):
        rm = grid_inp.shape[0] - batch_size*i
        if rm< batch_size:
            grid_batch = np.zeros((batch_size, 2))
            grid_batch[0:rm] = grid_inp[i*batch_size:i*batch_size+batch_size]
        else:
            grid_batch = grid_inp[i*batch_size:i*batch_size+batch_size]
        print(rm, i*batch_size,i*batch_size+batch_size)
        mf[i*batch_size:i*batch_size+batch_size]  = session.run(rmetric.MF, {rmetric.z: grid_batch})
    
    # now we clip the overflow away 
    mf = mf[:grid_inp_shape]
    print("mf grid", mf.shape)
    print(xx.shape, yy.shape, mf.shape)
    # Put the result into a color plot
    Z = mf.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap="OrRd")

    #plot actuals
    #plt.scatter(x_test[:,0], x_test[:,1],  )
    plt.scatter(x_test_encoded[:,0], x_test_encoded[:,1],  )
    plt.title("Full mf showing how inp space will change ")
    plt.colorbar()
    plt.show()