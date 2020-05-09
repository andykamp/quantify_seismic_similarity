''' This module evaluates the performance of a trained CPC encoder '''
# Path hack.
import sys, os
sys.path.insert(0, os.path.abspath('..'))
from scripts.data_utils.generators import SeismicGenerator, SequencePathGenerator
from scripts.data_utils.grid_utils import blockshaped, unblockshaped, plot_embeddings

from cpc.cluster import K_MEANS, T_SNE, scatter
from cpc.train_model_seimsic import CPCLayer

from os.path import join, basename, dirname, exists
import keras
from keras.models import Model
import numpy as np
import matplotlib
#matplotlib.use('PS')
#matplotlib.use(
#    "TkAgg"
#)  # suggested to avoid AttributeError: 'FigureCanvasMac' object has no attribute 'renderer'
import matplotlib.pyplot as plt
import cv2
# Importing seaborn to make nice plots.
import seaborn as sns
import pandas as pd
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

def build_model(encoder_path, image_shape, learning_rate):

    # Read the encoder
    encoder = keras.models.load_model(encoder_path)

    # Freeze weights
    encoder.trainable = False
    for layer in encoder.layers:
        layer.trainable = False

    # Define the classifier
    x_input = keras.layers.Input(image_shape)
    x = encoder(x_input)
    x = keras.layers.Dense(units=128, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(units=10, activation='softmax')(x)

    # Model
    model = keras.models.Model(inputs=x_input, outputs=x)

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    model.summary()

    return model


def benchmark_model(encoder_path, epochs, batch_size, output_dir, lr=1e-4, patch_size=64, color=False):

    # Prepare data
    train_data = SequencePathGenerator(batch_size, subset='train', patch_size=patch_size, color=color, rescale=True)

    validation_data = SequencePathGenerator(batch_size, subset='valid', patch_size=patch_size, color=color, rescale=True)

    # Prepares the model
    model = build_model(encoder_path, image_shape=(patch_size, patch_size, 3), learning_rate=lr)

    # Callbacks
    callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1/3, patience=2, min_lr=1e-4)]

    # Trains the model
    model.fit_generator(
        generator=train_data,
        steps_per_epoch=len(train_data),
        validation_data=validation_data,
        validation_steps=len(validation_data),
        epochs=epochs,
        verbose=1,
        callbacks=callbacks
    )

    # Saves the model
    model.save(join(output_dir, 'supervised.h5'))

def encode(encoder_path, encoder_weight_path ):

    encoder = load_encoder(
        encoder_path,
        encoder_weight_path,
    )
    # get batchSize* 7*7 patches
    pred_generator = SeismicGenerator(batch_size=1, subset='test')

    #prediction = encoder.predict_generator(pred_generator, steps=1)
    #print(prediction.shape)

    # calculate diff 
    #abs_feature = np.zeros(prediction.shape[0])
    #print("abs_feature", abs_feature.shape, prediction[0].shape)
    #distance = cv2.norm(prediction[0], prediction[1])
    #print("distance",distance )
    #distances = np.empty(128)
    #for i in range(prediction.shape[0]):
    #    distance = cv2.norm(prediction[0], prediction[i])
    #    print("distance", distance)
    #    distances[i] = distance
    #plt.imshow(np.resize(distances,(11*11)).reshape(11,11))
    #plt.show()    



    # Using for loop to print out siemsc form generator
    for item in pred_generator:
        print("item", item[0].shape)
        plot_embeddings(item[0], 64, 32, 256, stitched=True, channels=3)
        plot_embeddings(item[0].reshape(item[0].shape[0], item[0].shape[1]*item[0].shape[2],item[0].shape[3]), 64, 32, 256, channels=3)
        prediction = encoder.predict(item[0])
        print("prediction".prediciton.shape, item.shape)
        #plot_embeddings(prediction, 32)
        break
    

    #n_clusters = 10
    # generate clusters 
    #Y, z=K_MEANS(prediction, n_clusters)
    #print("k_mean:", z)


    # generate t-sne projections of each embedding 
    #pred_df = pd.DataFrame(prediction)
    #print(pred_df)
    digits_proj = T_SNE(prediction)
    print(digits_proj.shape)
    print(digits_proj)
    palette = np.array(sns.color_palette("GnBu_d", 49))
    print(palette)
    scatter(digits_proj, np.arange(49), 49, "GnBu_d", True)
    plt.show()

    #print(list(range(0,n_clusters)))
    #sns.palplot(np.array(sns.color_palette("hls", n_clusters)))
    #plt.imshow('digits_tsne-generated_18_cluster.png', dpi=120)
    #plt.show()


def load_encoder(encoder_path, encoder_weight_path):

    # Read and load the encoder model
    encoder = keras.models.load_model(encoder_path)
    # load weights into new model
    encoder.load_weights(encoder_weight_path)
    # show summary 
    encoder.summary()
    return encoder
def load_context_encoder(cpc_model_path, cpc_model_weight_path):
    # fetch the full model 
    model_old = load_cpc(cpc_model_path, cpc_model_weight_path)
    # Explicitly define new model input and output by slicing out old model layers
    model_new = Model(input=model_old.layers[0].input, 
                    output=model_old.layers[2].output)

    # Compile model to inspect
    model_new.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    # Visually inspect new model to confirm it the correct architecture
    model_new.summary()

def load_cpc(cpc_model_path, cpc_model_weight_path):

    # Read and load the encoder model
    cpc_model = keras.models.load_model(cpc_model_path, {"CPCLayer":CPCLayer})
    # load weights into new model
    cpc_model.load_weights(cpc_model_weight_path)
    # show summary 
    cpc_model.summary()
    return cpc_model

if __name__ == "__main__":
    ### from main directory python3 cpc/benchmark_model_seismic.py

    #load_encoder(
    #    encoder_path='models/seismic_64x64/encoder_seismic.h5',
    #    encoder_weight_path='models/seismic_64x64/encoder_seismic_weights.h5',
    #)
    #load_cpc(
    #    cpc_model_path='models/seismic_64x64/cpc_seismic.h5',
    #    cpc_model_weight_path='models/seismic_64x64/cpc_seismic_weights.h5',
    #)

    encode(
         encoder_path='models/cpc/seismic_64x64/encoder_seismic.h5',
         encoder_weight_path='models/cpc/seismic_64x64/encoder_seismic_weights.h5',
    )
