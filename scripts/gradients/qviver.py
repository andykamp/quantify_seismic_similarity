import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fname='/Users/anderskampenes/Documents/Dokumenter/NTNU/MASTER/code/seismic.png'
im = mpimg.imread(fname)
flat_image=(im[:,:,0]+im[:,:,1]+im[:,:,2])/3.

r, c = np.shape(flat_image)
gd = 4

test_slice = flat_image[::gd,::gd]  # sampling

fig,ax=plt.subplots(1,1)
the_image = ax.imshow(
                flat_image,
                zorder=0,alpha=1.0,
                cmap="Greys_r",
                origin="upper",
                interpolation="hermite",
            )
plt.colorbar(the_image)            
Y, X = np.mgrid[0:r:gd, 0:c:gd]
dY, dX = np.gradient(test_slice)
ax.quiver(X, Y, dX, dY, color='r')

#plt.show()
plt.savefig("qviver_gradient_arrows")