from minisom import MiniSom
import numpy as np
import matplotlib
matplotlib.use(
    "TkAgg"
)  # suggested to avoid AttributeError: 'FigureCanvasMac' object has no attribute 'renderer'
import matplotlib.pyplot as plt
#%matplotlib inline # used in order to see plot in notebook


# show seimsic image 

# get data to use 
data_cube = np.load("/Users/anderskampenes/Documents/Dokumenter/NTNU/MASTER/code/data/raw/f3-benchmark/test_once/test1_seismic.npy")
# make sure it is fomrated correctly 
img = data_cube[:,0,:].T
#plt.imshow(img)
#plt.show()
pixels = np.reshape(img, (img.shape[0]*img.shape[1], 1))


# initialize a 6-by-6 SOM with a learning rate of 0.5.
som = MiniSom(x= 2, y = 2, input_len = 1, sigma=0.1, learning_rate=0.2)
som.random_weights_init(pixels)

# save init weight for later visualization 
starting_weights = som.get_weights().copy()

#Then we train the SOM on 100 iterations.
som.train_random(pixels, 100)

#  quantize each pixel of the image
qnt = som.quantization(pixels)

print("mlknbbkjbkjbjkbjk")
# building new image
clustered = np.zeros(img.shape)
for i, q in enumerate(qnt):
  clustered[np.unravel_index(i, shape=(img.shape[0], img.shape[1]))] = q

print("done unraveling", img.shape, clustered.shape,starting_weights.shape,  som.get_weights().shape)

plt.figure(figsize=(12, 6))
plt.subplot(221)
plt.title('Original')
plt.imshow(img)
plt.subplot(222)
plt.title('Result')
plt.imshow(clustered)


plt.subplot(223)
plt.title('Initial Colors')
plt.imshow(np.squeeze(starting_weights))
plt.subplot(224)
plt.title('Learnt Colors')
plt.imshow(np.squeeze(som.get_weights()))

plt.tight_layout()
plt.show()