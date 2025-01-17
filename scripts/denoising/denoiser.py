import matplotlib.pyplot as plt

from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float
from skimage.util import random_noise
import cv2
import numpy as np


# DENOISING A PICTURE EXAMPLES FROM  https://scikit-image.org/docs/dev/auto_examples/filters/plot_denoise.html#sphx-glr-auto-examples-filters-plot-denoise-py

img = cv2.imread('/Users/anderskampenes/Documents/Dokumenter/NTNU/MASTER/code/seismic.png', 0)

#plt.imshow(img)
#plt.show()

original = img_as_float(img)

sigma = 0.155
noisy = random_noise(original, var=sigma**2)

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(8, 5),
                       sharex=True, sharey=True)

plt.gray()

# Estimate the average noise standard deviation across color channels.
sigma_est = estimate_sigma(noisy, multichannel=False, average_sigmas=True)
# Due to clipping in random_noise, the estimate will be a bit smaller than the
# specified sigma.
print("Estimated Gaussian noise standard deviation = {}".format(sigma_est))

ax[0, 0].imshow(original)
ax[0, 0].axis('off')
ax[0, 0].set_title('original')
ax[0, 1].imshow(denoise_tv_chambolle(original, weight=0.1, multichannel=False))
ax[0, 1].axis('off')
ax[0, 1].set_title('TV')
ax[0, 2].imshow(denoise_bilateral(original, sigma_color=0.05, sigma_spatial=15,
                multichannel=False))
ax[0, 2].axis('off')
ax[0, 2].set_title('Bilateral')
ax[0, 3].imshow(denoise_wavelet(original, multichannel=False))
ax[0, 3].axis('off')
ax[0, 3].set_title('Wavelet denoising')

ax[1, 1].imshow(denoise_tv_chambolle(original, weight=0.2, multichannel=False))
ax[1, 1].axis('off')
ax[1, 1].set_title('(more) TV')
ax[1, 2].imshow(denoise_bilateral(original, sigma_color=0.1, sigma_spatial=15,
                multichannel=False))
ax[1, 2].axis('off')
ax[1, 2].set_title('(more) Bilateral')
ax[1, 3].imshow(denoise_wavelet(original, multichannel=False, convert2ycbcr=False,
                                rescale_sigma=True))
ax[1, 3].axis('off')
ax[1, 3].set_title('Wavelet denoising\nin YCbCr colorspace')
ax[1, 0].imshow(original)
ax[1, 0].axis('off')
ax[1, 0].set_title('Original')

fig.tight_layout()

#plt.show()
plt.savefig('1.removing_original_noise.jpg')
