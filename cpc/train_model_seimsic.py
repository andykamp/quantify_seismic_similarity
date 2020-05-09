'''
This module describes the contrastive predictive coding model from DeepMind:

Oord, Aaron van den, Yazhe Li, and Oriol Vinyals.
"Representation Learning with Contrastive Predictive Coding."
arXiv preprint arXiv:1807.03748 (2018).
'''
# Path hack.
import sys, os
sys.path.insert(0, os.path.abspath('..'))
from scripts.data_utils.generators import SeismicGenerator, SequencePathGenerator
from os.path import join, basename, dirname, exists
import keras
from keras import backend as K
from pathlib import Path

# declare parent dir name 
dirname = sys.path[0] # parent directory

def network_encoder(x, code_size):

    ''' Define the network mapping images to embeddings '''

    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=256, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(units=code_size, activation='linear', name='encoder_embedding')(x)

    return x


def network_autoregressive(x):

    ''' Define the network that integrates information along the sequence '''

    # x = keras.layers.GRU(units=256, return_sequences=True)(x)
    # x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GRU(units=256, return_sequences=False, name='ar_context')(x)

    return x


def network_prediction(context, code_size, predict_terms):

    ''' Define the network mapping context to multiple embeddings '''

    outputs = []
    for i in range(predict_terms):
        outputs.append(keras.layers.Dense(units=code_size, activation="linear", name='z_t_{i}'.format(i=i))(context))

    if len(outputs) == 1:
        output = keras.layers.Lambda(lambda x: K.expand_dims(x, axis=1))(outputs[0])
    else:
        output = keras.layers.Lambda(lambda x: K.stack(x, axis=1))(outputs)

    return output


class CPCLayer(keras.layers.Layer):

    ''' Computes dot product between true and predicted embedding vectors '''

    def __init__(self, **kwargs):
        super(CPCLayer, self).__init__(**kwargs)

    def call(self, inputs):

        # Compute dot product among vectors
        preds, y_encoded = inputs
        dot_product = K.mean(y_encoded * preds, axis=-1)
        dot_product = K.mean(dot_product, axis=-1, keepdims=True)  # average along the temporal dimension

        # Keras loss functions take probabilities
        dot_product_probs = K.sigmoid(dot_product)

        return dot_product_probs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)


def network_cpc(image_shape, terms, predict_terms, code_size, learning_rate):

    ''' Define the CPC network combining encoder and autoregressive model '''

    # Set learning phase (https://stackoverflow.com/questions/42969779/keras-error-you-must-feed-a-value-for-placeholder-tensor-bidirectional-1-keras)
    K.set_learning_phase(1)

    # Define encoder model
    encoder_input = keras.layers.Input(image_shape)
    encoder_output = network_encoder(encoder_input, code_size)
    encoder_model = keras.models.Model(encoder_input, encoder_output, name='encoder')
    encoder_model.summary()

    # Define rest of model
    x_input = keras.layers.Input((terms, image_shape[0], image_shape[1], image_shape[2]))
    x_encoded = keras.layers.TimeDistributed(encoder_model)(x_input) # applies the encoder to teach timestep
    context = network_autoregressive(x_encoded)
    # this output is used as the context for the predictions
    preds = network_prediction(context, code_size, predict_terms)

    y_input = keras.layers.Input((predict_terms, image_shape[0], image_shape[1], image_shape[2]))
    y_encoded = keras.layers.TimeDistributed(encoder_model)(y_input) # applies the encoder to teach timestep

    # Loss
    dot_product_probs = CPCLayer()([preds, y_encoded])

    # Model
    cpc_model = keras.models.Model(inputs=[x_input, y_input], outputs=dot_product_probs)

    # Compile model
    cpc_model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
    cpc_model.summary()

    return cpc_model


def train_model(epochs, batch_size, output_dir, code_size, lr=1e-4, terms=3, predict_terms=4, image_size=256,patch_size=64, stride=32, num_crops=10, augmentation=True, sequence=True, verbose=False, steps_per_epoch=None, validation_steps=None):

    # create output folder if it does not exist
    if not os.path.exists(output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Prepare data
    train_data = SeismicGenerator(batch_size=batch_size, subset='train', 
                                       positive_samples=batch_size // 2, predict_terms=predict_terms,
                                       image_size=image_size, patch_size=patch_size, stride=stride,num_crops=num_crops, augmentation=augmentation, sequence=sequence, verbose=verbose)

    validation_data = SeismicGenerator(batch_size=batch_size, subset='valid',
                                            positive_samples=batch_size // 2, predict_terms=predict_terms,
                                            image_size=image_size, patch_size=patch_size, stride=stride,num_crops=num_crops, augmentation=augmentation, sequence=sequence, verbose=verbose)

    # Prepares the model
    model = network_cpc(image_shape=(patch_size, patch_size, 3), terms=terms, predict_terms=predict_terms, 
                        code_size=code_size, learning_rate=lr) # CHANGED CHANNELS FROM 3 TO 1 

    # Callbacks
    callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1/3, patience=2, min_lr=1e-4), keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3,  mode='auto'), keras.callbacks.ModelCheckpoint(join(output_dir, 'checkpoint.h5'), monitor='val_loss', verbose=1, save_best_only=True, mode='auto')]

    # override  default steps if spesifies     
    if steps_per_epoch is None: steps_per_epoch = len(train_data)
    if validation_steps is None: validation_steps = len(validation_data)
    print("steps_per_epoch", steps_per_epoch)
    print("validation_steps", validation_steps)

    # Trains the model
    model.fit_generator(
        generator=train_data,
        steps_per_epoch= steps_per_epoch,
        validation_data=validation_data,
        validation_steps=validation_steps,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks
    )

    # Saves the model
    # Remember to add custom_objects={'CPCLayer': CPCLayer} to load_model when loading from disk
    model.save(join(output_dir, 'cpc_seismic.h5'))
    model.save_weights(join(output_dir, "cpc_seismic_weights.h5"))

    # Saves the encoder alone
    encoder = model.layers[1].layer
    encoder.save(join(output_dir, 'encoder_seismic.h5'))
    encoder.save_weights(join(output_dir, "encoder_seismic_weights.h5"))


def dep_train_model(epochs, batch_size, output_dir, code_size, lr=1e-4, terms=3, predict_terms=4, patch_size=64, color=False):

    # Prepare data
    train_data = SequencePathGenerator(batch_size=batch_size, subset='train', terms=terms,
                                       positive_samples=batch_size // 2, predict_terms=predict_terms,
                                       patch_size=patch_size, color=color, rescale=True)

    validation_data = SequencePathGenerator(batch_size=batch_size, subset='valid', terms=terms,
                                            positive_samples=batch_size // 2, predict_terms=predict_terms,
                                            patch_size=patch_size, color=color, rescale=True)

    # Prepares the model
    model = network_cpc(image_shape=(patch_size, patch_size, 3), terms=terms, predict_terms=predict_terms, 
                        code_size=code_size, learning_rate=lr) # CHANGED CHANNELS FROM 3 TO 1 

    # Callbacks
    callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1/3, patience=2, min_lr=1e-4), keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3,  mode='auto')]

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
    # Remember to add custom_objects={'CPCLayer': CPCLayer} to load_model when loading from disk
    model.save(join(output_dir, 'cpc_seismic.h5'))
    model.save_weights(join(output_dir, "cpc_seismic_weights.h5"))

    # Saves the encoder alone
    encoder = model.layers[1].layer
    encoder.save(join(output_dir, 'encoder_seismic.h5'))
    encoder.save_weights(join(output_dir, "encoder_seismic_weights.h5"))


if __name__ == "__main__":

    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    print(timestr)
    train_model(
            epochs=5,
            batch_size=1,
            output_dir=join(dirname, 'models/cpc/seismic_64x64_aug'),
            code_size=1024,
            lr=1e-3,
            terms=3,
            predict_terms=4,
            patch_size=64,
        )
    print("Done training")

