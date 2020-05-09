''' This module contains code to handle data '''

import os
import numpy as np
import scipy.ndimage
from PIL import Image
import scipy
import sys
import matplotlib
#matplotlib.use('PS')
#matplotlib.use(
#    "TkAgg"
#)  # suggested to avoid AttributeError: 'FigureCanvasMac' object has no attribute 'renderer'
import matplotlib.pyplot as plt
from skimage.util import view_as_blocks
from sklearn.feature_extraction import image
import tensorflow as tf

"""
Convert a seismic cube into  a series of sentences usabel for CPC 
The article descibes the following pattern:

Pre-processing:
-(own) from a random iline/xline/tline > 256*256 we extract pathces of 256x256 using sklearn image.extract_patches_2d
-from a 256x256 image we extract a 7x7 grid of 64x64 crops
with 32 pixels overlap. 

-(TODO)Simple data augmentation proved helpful on both the 256x256 images and the
64x64 crops. 
-(DONE)The 256x256 images are randomly cropped from a 300x300 image
- horizontally flipped with a probability of 50% and converted to greyscale. 
-For each of the 64x64 crops we randomly take
a 60x60 subcrop and pad them back to a 64x64 image.
-MNIST example uses RGB channels. Seimsic is grayscale. Would it help to pick and generate x attributes as channels
Encoder:
-(DONE)Each crop is then encoded by the ResNet-v2-101 encoder
- (DONE)We use the outputs from the third residual
block, and spatially mean-pool to get a single 1024-d vector per 64x64 patch. This results in a
7x7x1024 tensor.
"""
class SeismicHandler(object):

    ''' Provides a convenient interface to manipulate MNIST data '''

    def __init__(self):

        # Download data if needed
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.load_dataset()

        # Load Lena image to memory
        self.lena = Image.open('resources/lena.jpg')

    def load_dataset(self):
        # Credit for this function: https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py

        # We first define a download function, supporting both Python 2 and 3.
        if sys.version_info[0] == 2:
            from urllib import urlretrieve
        else:
            from urllib.request import urlretrieve
        
        # downloads file remotely if it does not exist locallt
        def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
            print("Downloading %s" % filename)
            urlretrieve(source + filename, filename)

        # generates training data both in inline and xline wiewpoints
        def generate_training_data(filename):
            print("Generating training data...")
            # get data to use 
            filename = "/Users/anderskampenes/Documents/Dokumenter/NTNU/MASTER/code/contrastive-predictive-coding/resources/256*256-data-f3-iline.npy"
            print("checking if file exist", os.path.exists(filename))
            if not os.path.exists(filename):
                seismic = np.load("/Users/anderskampenes/Downloads/f3_entire_int8.npy")#"/Users/anderskampenes/Documents/Dokumenter/NTNU/MASTER/code/data/raw/f3-benchmark/test_once/test1_seismic.npy")
                # transpose it so tline is upwards
                seismic = seismic.T
                print("Seimsic data shape", seismic.shape)
                crop_size = 256
                num_crops_per_slice = 10
                num_total_crops = (seismic.shape[0])*num_crops_per_slice
                #  num_total_crops = (seismic.shape[0]+seismic.shape[1]+seismic.shape[2])*num_crops_per_slice

                print("num_total_crops", num_total_crops)
                # allocate the training data array
                data = np.empty([num_total_crops, crop_size, crop_size])
                # for each direction in the seismic, we crop x 256*256 images from each seismic slice
                # we start in  
                for i in range(seismic.shape[0]):
                    patches = image.extract_patches_2d( seismic[i,:,:], (crop_size, crop_size), num_crops_per_slice)
                    data[i*num_crops_per_slice:i*num_crops_per_slice+num_crops_per_slice,:,:] = patches
                    patches = None
                print("done with inline")

                #for ii in range(seismic.shape[1]):
                #    patches = image.extract_patches_2d( seismic[:,ii,:], (crop_size, crop_size), num_crops_per_slice)
                #    data[(i +ii)*num_crops_per_slice:(i + ii)*num_crops_per_slice+num_crops_per_slice,:,:] = patches
                #    patches = None
                #print("done with xline")
                #for iii in range(seismic.shape[2]):
                #    patches = image.extract_patches_2d( seismic[:,:,iii], (crop_size, crop_size), num_crops_per_slice)
                #    data[(i+ii+iii)*num_crops_per_slice:(i+ii+iii)*num_crops_per_slice+num_crops_per_slice,:,:] = patches
                #    patches = None
                #print("done with tline", (i+ii+iii)*num_crops_per_slice)
                #plt.imshow(data[20040])
                #plt.show()


                print("saving data..")
                np.save("resources/256*256-data-f3-iline", data)
                print("done saving data")
            else: 
                print("loading data..")
                data = np.load(filename)
                print("done loading data")
                print("data", data.shape)
                



            # convert loaded data to training set
            test_split = 0.1
            val_split = 0.1

            # shuffle the dataset

            #data = np.random.shuffle(data)
            
            X_train = data[:int(data.shape[0]*(1-test_split))]
            y_train = np.arange(0,int(data.shape[0]*(1-test_split)))
            X_test = data[ int(data.shape[0]*(1-test_split)):]
            y_test = np.arange(int(data.shape[0]*(1-test_split)), data.shape[0])
            print("TRAIN-TEST split: ", X_train.shape,  X_test.shape, y_train.shape, y_test.shape)
             # We reserve the last 10000 training examples for validation.
            X_train, X_val = X_train[ :int(X_train.shape[0]*(1-val_split))], X_train[ int(X_train.shape[0]*(1-val_split)):]
            y_train, y_val = np.arange(0,int(y_train.shape[0]*(1-val_split))), np.arange(int(y_train.shape[0]*(1-val_split)), y_train.shape[0])
            print("TRAIN-VAL split: ", X_train.shape,  X_val.shape)

            print("Sequence of indexes used as labels for train, val and test: ",  y_train[-1], y_val[-1], y_test[-1])



            
            return X_train, y_train, X_val, y_val, X_test, y_test


        # We then define functions for loading seimisc images and infer labels (indexes).
        def load_seismic_data(filename):
            if not os.path.exists(filename):
                #download(filename)
                return generate_training_data(filename)

        # We can now download and read the training and test set images and labels.
        # We use indexes as labels as seimsic is continues and gives some supervision
        X_train, y_train, X_val, y_val, X_test, y_test = load_seismic_data('resources/f3')
        #plt.imshow(X_train[0,:,:])
        #plt.show()
        # We just return all the arrays in order, as expected in main().
        # (It doesn't matter how we do this as long as we can read them again.)
        return X_train, y_train, X_val, y_val, X_test, y_test
       
    

    def process_batch(self, batch, batch_size, image_size, patch_size, color=False, rescale=True):
    
        #batch = batch.reshape((batch_size, 1, image_size, image_size))
        # convert the greyscale to rgb by repeating the greyscal 3 times 
        batch = np.repeat(batch[..., np.newaxis], 3, -1)
        #batch = np.concatenate([batch, batch, batch], axis=1)
        #print(batch.shape)
        # Modify images if color distribution requested
        if color:

            # Binarize images
            batch[batch >= 0.5] = 1
            batch[batch < 0.5] = 0

            # For each image in the mini batch
            for i in range(batch_size):

                # Take a random crop of the Lena image (background)
                x_c = np.random.randint(0, self.lena.size[0] - image_size)
                y_c = np.random.randint(0, self.lena.size[1] - image_size)
                image = self.lena.crop((x_c, y_c, x_c + image_size, y_c + image_size))
                image = np.asarray(image).transpose((2, 0, 1)) / 255.0

                # Randomly alter the color distribution of the crop
                for j in range(3):
                    image[j, :, :] = (image[j, :, :] + np.random.uniform(0, 1)) / 2.0

                # Invert the color of pixels where there is a number
                image[batch[i, :, :, :] == 1] = 1 - image[batch[i, :, :, :] == 1]
                batch[i, :, :, :] = image

        # Rescale to range [-1, +1]
        if rescale:
            batch = batch * 2 - 1

        # Channel last
        #batch = batch.transpose((0, 2, 3, 1)

        # make patches from the image crop 
        n=7
        nn = n*n
        patches = []
        for img in range(batch_size):
            for i in range(n):
                for j in range(n):
                    patches.append(batch[ img, i*32:i*32+64, j*32:j*32+64,:])
        batch = np.asarray(patches)

        #print("plotting batches...", batch.shape)
        #plot_patches(batch[0:nn,:,:])

        return batch


    def get_batch(self, subset, batch_size, image_size=256, patch_size=64, color=False, rescale=True):

        # Select a subset
        if subset == 'train':
            X = self.X_train
            y = self.y_train
        elif subset == 'valid':
            X = self.X_val
            y = self.y_val
        elif subset == 'test':
            X = self.X_test
            y = self.y_test

         # Random choice of samples
        idx = np.random.choice(X.shape[0], batch_size)
        batch = X[idx, :, :]
        # we now how our selected batch and want to to create patches  on the fly for each batch_size of them

        # Process batch
        batch = self.process_batch(batch, batch_size, image_size, patch_size, color, rescale)

        # Image label
        labels = y[idx]

        return batch.astype('float32'), labels.astype('int32')


    def get_n_samples(self, subset):

        if subset == 'train':
            y_len = self.y_train.shape[0]
        elif subset == 'valid':
            y_len = self.y_val.shape[0]
        elif subset == 'test':
            y_len = self.y_test.shape[0]

        return y_len


class SeismicGenerator(object):

    ''' Data generator providing MNIST data '''

    def __init__(self, batch_size, subset, image_size=256, patch_size=64, color=False, rescale=True):

        # Set params
        self.batch_size = batch_size
        self.subset = subset
        self.image_size = image_size
        self.patch_size= patch_size
        self.color = color
        self.rescale = rescale

        # Initialize MNIST dataset
        self.seismic_handler = SeismicHandler()
        self.n_samples = self.seismic_handler.get_n_samples(subset)
        self.n_batches = self.n_samples // batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.n_batches

    def next(self):

        # Get data
        x, y = self.seismic_handler.get_batch(self.subset, self.batch_size, self.image_size, self.patch_size, self.color, self.rescale)
        # TODO  Convert y to one-hot 
        #y_h = np.eye(10)[y]

        return x, y


class SequencePathGenerator(object):

    ''' Data generator providing lists of sorted numbers '''

    def __init__(self, batch_size, subset, terms, positive_samples=1, predict_terms=1, image_size=256,patch_size=64, color=False, rescale=True):

        # Set params
        self.positive_samples = positive_samples
        self.predict_terms = predict_terms
        self.batch_size = batch_size
        self.subset = subset
        self.terms = terms
        self.image_size = image_size
        self.patch_size= patch_size
        self.color = color
        self.rescale = rescale

        # Initialize MNIST dataset
        self.seismic_handler = SeismicHandler()
        self.n_samples = self.seismic_handler.get_n_samples(subset) // terms
        self.n_batches = self.n_samples // batch_size
        
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.n_batches

    def next(self):

        # get batch of [batch_size*grid, image_size, image_size, 3]
        x, _ = self.seismic_handler.get_batch(self.subset, self.batch_size, self.image_size, self.patch_size, self.color, self.rescale)
        # extract the rows to predict z+x
        # the rows for one image is every 7th row 7 times
        sequence = np.empty([self.batch_size, 7, x.shape[1], x.shape[2], x.shape[3]])
        for i in range(self.batch_size):
            sequence[i]= x[i*49:i*49+49:7]
            #plot_patches_column(sequence[i], self.terms)
        # Assemble batch
        x_images = sequence[:, :-self.predict_terms, ...]
        y_images = sequence[:, -self.predict_terms:, ...]

        # set random labels TODO do i need it 
        sentence_labels = np.ones((self.batch_size, 1)).astype('int32')
        # Randomize
        #idxs = np.random.choice(sentence_labels.shape[0], sentence_labels.shape[0], replace=False)
        return [x_images, y_images], sentence_labels #[x_images[idxs, ...], y_images[idxs, ...]], sentence_labels[idxs, ...]


        #idxs = np.random.choice(sentence_labels.shape[0], sentence_labels.shape[0], replace=False)
        return [x_images, y_images], sentence_labels #[x_images[idxs, ...], y_images[idxs, ...]], sentence_labels[idxs, ...]



def plot_patches(patches):
    num_y, num_x =7,7
    ''' Draws a plot to stich together patches to a image '''
    counter = 1
    for y in range(num_y):
        for x in range(num_x):
            plt.subplot(num_y, num_x, counter)
            plt.imshow(patches[counter-1])
            plt.axis('off')
            counter += 1
    plt.show()

def plot_patches_column(patches, terms):
    n = patches.shape[0]

    ''' Draws a plot to stich together patches in one column '''
    counter = 1
    for i in range(n): 
        plt.subplot(n, 1, counter)
        plt.imshow(patches[i])
        if i >= terms:
            plt.imshow(patches[i],cmap=plt.cm.RdBu)
        else:
            plt.imshow(patches[i])
        plt.axis('off')
        counter += 1
    plt.show()


if __name__ == "__main__":
    # test seimsic handler and loader
    #a = SeismicHandler()
    #ag = SeismicGenerator(batch_size=1, subset='train')
    #for val in ag:
    #    print(val)
    #    break

    # test GridPatchGenerator 
    #ag = SeismicGenerator(batch_size=2, subset='train', color=False, rescale=False)
    #for images in ag:
    #    print("GridPatchGenerator: ", images.shape)
    #    #plot_sequences(x, y, labels, output_path=r'resources/batch_sample_sorted.png')
    #    break


    # Test SequencePathGenerator
    ag = SequencePathGenerator(batch_size=10, subset='train', terms=4, positive_samples=4, predict_terms=4,)
    for (x, y), labels in ag:
        print("generator, x, y, labels", x.shape, y.shape, labels)
        #plot_sequences(x, y, labels, output_path=r'resources/batch_sample_sorted.png')
        break

    