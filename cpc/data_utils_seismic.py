''' This module contains code to handle data '''

import os
import numpy as np
import scipy.ndimage
from PIL import Image
import scipy
import sys
import matplotlib
#matplotlib.use('PS')
matplotlib.use(
    "TkAgg"
)  # suggested to avoid AttributeError: 'FigureCanvasMac' object has no attribute 'renderer'
import matplotlib.pyplot as plt
from skimage.util import view_as_blocks
"""
Convert a seismic cube into  a series of sentences usabel for CPC 
The article descibes the following pattern:

Pre-processing:
-from a 256x256 image we extract a 7x7 grid of 64x64 crops
with 32 pixels overlap. 
-(TODO)Simple data augmentation proved helpful on both the 256x256 images and the
64x64 crops. 
-(TODO)The 256x256 images are randomly cropped from a 300x300 image, horizontally flipped
with a probability of 50% and converted to greyscale. 
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
            data = np.load("/Users/anderskampenes/Downloads/f3_entire_int8.npy")#"/Users/anderskampenes/Documents/Dokumenter/NTNU/MASTER/code/data/raw/f3-benchmark/test_once/test1_seismic.npy")
            # transpose it so tline is upwards
            data = data.T
            print("Seimsic data shape", data.shape)
            test_split = 0.1
            val_split = 0.1

            # we first add in xline direction
            X_train = data[:, :int(data.shape[1]*(1-test_split)),:]
            y_train = np.arange(0,int(data.shape[1]*(1-test_split)))
            X_test = data[:, int(data.shape[1]*(1-test_split)):,:]
            y_test = np.arange(int(data.shape[1]*(1-test_split)), data.shape[1])
            print("TRAIN-TEST split: ", X_train.shape,  X_test.shape, y_train.shape, y_test.shape)
             # We reserve the last 10000 training examples for validation.
            X_train, X_val = X_train[:, :int(X_train.shape[1]*(1-val_split)),:], X_train[:, int(X_train.shape[1]*(1-val_split)):,:]
            y_train, y_val = np.arange(0,int(y_train.shape[0]*(1-val_split))), np.arange(int(y_train.shape[0]*(1-val_split)), y_train.shape[0])
            print("TRAIN-VAL split: ", X_train.shape,  X_val.shape)

            print("Sequence of indexes used as labels for train, val and test: ",  y_train[-1], y_val[-1], y_test[-1])
            #return X_train, y_train, X_val, y_val, X_test, y_test
            return np.transpose(X_train,(1,0,2)), y_train, np.transpose(X_val,(1,0,2)), y_val, np.transpose(X_test,(1,0,2)), y_test


            #np.save("resources/train-data-f3", X_train)
            #np.save("resources/train-data-f3", y_train)
            #np.save("resources/test-data-f3", X_test)
            #np.save("resources/test-data-f3", y_test)

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
       
    

    def process_batch(self, batch, batch_size, image_size, color=False, rescale=True):
        def generate_patches_from_slice(seismic_slice, image_size=64):
            """ Generate fixed number of pathes for a seimisc slice. Each patch should have the same center cordinate in the direction they are sampled from to form a sequence of patches """
            ### TODO add support for overlap 

            
            
            def flatten_patches(patches, image_size):
                num_y, num_x = patches.shape[0], patches.shape[1]
                flatten = np.empty([num_y*num_x, image_size*image_size])

                ''' Draws a plot to stich together patches to a image '''
                i = 0
                for y in range(num_y):
                    for x in range(num_x):
                        print("flattennned", patches[y,x].shape,  patches[y,x].flatten().shape)
                        flatten[i,:] = patches[y,x].flatten()
                        i=i+1
                return flatten

            
            # infer how much padding is needed to be bale to extract patches (with no overlap)
            pad_x = image_size - seismic_slice.shape[0] % image_size
            pad_y = image_size -  seismic_slice.shape[-1] % image_size
            #print("inferred paddings", pad_x, pad_y)
        
            padded_slice = np.pad(seismic_slice, ((0,pad_x),(0, pad_y)), 'edge')
            #print("padded__slice", padded_slice.shape)
            patches = view_as_blocks(padded_slice, (image_size, image_size))
            #print("patches", patches.shape)
            # check if pathces for whole image is correct
            #plot_patches(patches)
            # check if patches for a row is correct
            #plot_patches_column(patches)
            #flattened_patches = flatten_patches(patches, image_size)
            #checking that flattend arrays are correct
            #print(flattened_patches.shape,)
            #plt.imshow(flattened_patches[2,:].reshape(64,64))
            #plt.show()

            # TODO format patches to (channel, height, width)
            return patches # return a (row,col, image_size*image_size) array
            #return flattened_patches



        # hardcode the number of patches in each direction 
        num_rows, num_cols = 8,11
        newBatch = np.empty([batch_size, num_rows, num_cols, image_size, image_size])
        # For each image in the mini batch
        for i in range(batch_size):
            newBatch[i,:,:,:,:] = generate_patches_from_slice(batch[i,:,:], image_size)

        # validate the patch byy verbose
        #print("newBatch",newBatch.shape)
        #plt.imshow(newBatch[0,0,1])
        #plt.show()

        # Channel last
        newBatch = newBatch.reshape((newBatch.shape[0], newBatch.shape[1], newBatch.shape[2], newBatch.shape[3], newBatch.shape[4], 1))
        return newBatch

    

    def get_batch(self, subset, batch_size, image_size=64, color=False, rescale=True):

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
        batch = self.process_batch(batch, batch_size, image_size, color, rescale)

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

    def __init__(self, batch_size, subset, image_size=64, color=False, rescale=True):

        # Set params
        self.batch_size = batch_size
        self.subset = subset
        self.image_size = image_size
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
        x, y = self.seismic_handler.get_batch(self.subset, self.batch_size, self.image_size, self.color, self.rescale)

        # Convert y to one-hot
        y_h = np.eye(10)[y]

        return x, y_h


class SequencePathGenerator(object):

    ''' Data generator providing lists of sorted numbers '''

    def __init__(self, batch_size, subset, terms, positive_samples=1, predict_terms=1, image_size=28, color=False, rescale=True):

        # Set params
        self.positive_samples = positive_samples
        self.predict_terms = predict_terms
        self.batch_size = batch_size
        self.subset = subset
        self.terms = terms
        self.image_size = image_size
        self.color = color
        self.rescale = rescale

        # Initialize MNIST dataset
        self.seismic_handler = SeismicHandler()
        self.n_samples = self.seismic_handler.get_n_samples(subset) // terms
        self.n_batches = self.n_samples // batch_size
        self.batch=None
        self.currentBatchColumn = 0
        
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.n_batches

    def next(self):

             

        # Retrieve batch of row*col gridded patches
        # each batch is on the form [batch_size, num_rows, num_cols, image_size, image_size]
        self.batch, _ = self.seismic_handler.get_batch(self.subset, self.batch_size, self.image_size, self.color, self.rescale)

        

        def plot_sentence(patches, col, terms):
            num_y, num_x = patches.shape[0], patches.shape[1]

            ''' Draws a plot to stich together patches in one column '''
            counter = 1
            for y in range(num_y): 
                plt.subplot(num_y, 1, counter)
                # plotdifferent color for the predictie terms of the sequence
                if y >= terms:
                    plt.imshow(patches[y,col],cmap=plt.cm.RdBu)
                else:
                    plt.imshow(patches[y,col])
                plt.axis('off')
                counter += 1
            plt.show()
        sentence = self.batch[:,:self.terms+ self.predict_terms, :,:,:] 
        

        # plot all columns for a given image (a batch item)
        #for i in range(sentence.shape[2]):
        #    plot_sentence(sentence[0,:,:,:,:,0], i, self.terms)
        
            
        #self.currentBatchColumn = self.currentBatchColumn+1

        
        # Assemble batch
        # hardcode to only use the first column for each image TODO FIX
        images = sentence[:,:,0,:,:,:]
        x_images = images[:, :-self.predict_terms, ...]
        y_images = images[:, -self.predict_terms:, ...]


        # set random labels TODO do i need it 
        sentence_labels = np.ones((self.batch_size, 1)).astype('int32')
        # Randomize
        #idxs = np.random.choice(sentence_labels.shape[0], sentence_labels.shape[0], replace=False)
        return [x_images, y_images], sentence_labels #[x_images[idxs, ...], y_images[idxs, ...]], sentence_labels[idxs, ...]


class GridPatchGenerator(object):

    ''' Data generator providing lists of sorted numbers '''

    def __init__(self, batch_size, subset, terms, positive_samples=1, predict_terms=1, image_size=28, color=False, rescale=True):

        # Set params
        # Set params
        self.positive_samples = positive_samples
        self.predict_terms = predict_terms
        self.batch_size = batch_size
        self.subset = subset
        self.terms = terms
        self.image_size = image_size
        self.color = color
        self.rescale = rescale

        # Initialize MNIST dataset
        self.seismic_handler = SeismicHandler()
        self.n_samples = self.seismic_handler.get_n_samples(subset) // terms
        self.n_batches = self.n_samples // batch_size
        self.batch=None
        
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.n_batches

    def next(self):

        # TODO DO I NEED TO ACCOUND FOR POSITIVE AND NEGATIVE SAMPLES? AS THIS IS UNSUPERVISED I GUESS I DO NOT?
        # COULD POSSIBLE MUIX DIFFERENT SLICE INDEXES TO GENERATE NEGATIVES??
       
        # Retrieve batch of row*col gridded patches
        #if self.batch is None:
        # each batch is on the form [batch_size, num_rows, num_cols, image_size, image_size]
        self.batch, _ = self.seismic_handler.get_batch(self.subset, self.batch_size, self.image_size, self.color, self.rescale)
        

        patches = self.batch
        print("patches", patches.shape)
        # check if pathces for whole image is correct
        #plot_patches(patches[0,:,:,:,:,0])
        #reshape by flattening patches underneath eachother so we have [tot_num_patches, 64, 64, 1]
        patches = patches.reshape((patches.shape[0]*patches.shape[1]*patches.shape[2], patches.shape[3], patches.shape[4], 1))
        print("patches shape", patches.shape)

        
        


        return patches

   

def plot_patches(patches):
    num_y, num_x = patches.shape[0], patches.shape[1]

    ''' Draws a plot to stich together patches to a image '''
    counter = 1
    for y in range(num_y):
        for x in range(num_x):
            plt.subplot(num_y, num_x, counter)
            plt.imshow(patches[y,x])
            plt.axis('off')
            counter += 1
    plt.show()

def plot_patches_column(patches, col=0):
    num_y, num_x = patches.shape[0], patches.shape[1]

    ''' Draws a plot to stich together patches in one column '''
    counter = 1
    for y in range(num_y): 
        plt.subplot(num_y, 1, counter)
        plt.imshow(patches[y,col])
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
    ag = GridPatchGenerator(batch_size=1, subset='train', terms=4, positive_samples=4, predict_terms=4, image_size=64, color=True, rescale=False)
    for images in ag:
        print("GridPatchGenerator: ", images.shape)
        #plot_sequences(x, y, labels, output_path=r'resources/batch_sample_sorted.png')
        break


    # Test SequencePathGenerator
    #ag = SequencePathGenerator(batch_size=10, subset='train', terms=4, positive_samples=4, predict_terms=4, image_size=64, color=True, rescale=False)
    #for (x, y), labels in ag:
    #    print("generator, x, y, labels", x.shape, y.shape, labels)
    #    #plot_sequences(x, y, labels, output_path=r'resources/batch_sample_sorted.png')
    #    break

    