import numpy as np
import matplotlib
matplotlib.use(
    "TkAgg"
)  # suggested to avoid AttributeError: 'FigureCanvasMac' object has no attribute 'renderer'
import matplotlib.pyplot as plt

def blockshaped(arr, nrows, ncols):
        """
        Return an array of shape (n, nrows, ncols) where
        n * nrows * ncols = arr.size

        If arr is a 2D array, the returned array looks like n subblocks with
        each subblock preserving the "physical" layout of arr.
        """
        h, w = arr.shape
        print("blockshaped", h, w, arr.shape, nrows, ncols)
        return (arr.reshape(h//nrows, nrows, -1, ncols)
                .swapaxes(1,2)
                .reshape(-1, nrows, ncols))


def unblockshaped(arr, h, w, channels=False):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    
    if channels:
        n, nrows, ncols, channels = arr.shape
        print("unblockshaped", n, nrows, ncols)
        return (arr.reshape(h//nrows, -1, nrows, ncols,channels)
                .swapaxes(1,2)
                .reshape(h, w,3))
    else:
        n, nrows, ncols = arr.shape
        print("unblockshaped", n, nrows, ncols)
        return (arr.reshape(h//nrows, -1, nrows, ncols)
                .swapaxes(1,2)
                .reshape(h, w))

def plot_grids(original, grids, alpha, filename):

    ''' Plots different poolings for subgroups'''
    for i in range(len(grids)):
        plt.subplot(1, len(grids), i+1)
        plt.imshow(original)
        plt.imshow(grids[i],alpha=alpha)
        #plt.axis('off')
    #plt.show()
    plt.savefig(filename)

def pool_grid(attr, grid_blocks_flattened, img_size, patch_size):
    grid = np.vstack([attr]* grid_blocks_flattened.shape[-1]) 
    grid = np.transpose(grid, (-1,0)).reshape(attr.shape[0], patch_size, patch_size)
    print(grid.shape)
    grid =  unblockshaped(grid, img_size,img_size)
    return grid

if __name__ == '__main__':
    #get data to u  se 
    data_cube = np.load("/Users/anderskampenes/Documents/Dokumenter/NTNU/MASTER/code/data/raw/f3-benchmark/test_once/test1_seismic.npy")
    # make sure it is fomrated correctly 
    img_size = 250
    img = data_cube[0,:img_size,:img_size].T # subcrop to 256*256
    print("img shape: ", img.shape)
    #plt.imshow(img)
    #plt.show()
    patch_size = 5
    patch_ratio = int(img_size /patch_size )
    print(img_size, patch_size, patch_ratio)
    grid_blocks = blockshaped(img, patch_size, patch_size)
    print(grid_blocks.shape)
    # validate first block 
    #plt.imshow(grid_blocks[0])
    #plt.show()

    # flatten block s to be able to multiply 
    grid_blocks_flattened = grid_blocks.reshape(grid_blocks.shape[0], -1)
    #get avg of each bloc for each row (subgrid)
    avgs = np.average(grid_blocks_flattened, axis=1)
    avgs_grid =  pool_grid(avgs, grid_blocks_flattened, img_size, patch_size)
    
    # get min of each bloc for each row (subgrid)
    mins = np.amin(grid_blocks_flattened, axis=1)
    mins_grid =  pool_grid(mins, grid_blocks_flattened, img_size, patch_size)
        
    # get max of each bloc  for each row (subgrid)
    maxs = np.amax(grid_blocks_flattened, axis=1)  
    maxs_grid =  pool_grid(maxs, grid_blocks_flattened, img_size, patch_size)
    
    # convert min to abs 
    abs_mins = np.absolute(mins)
    abs_mins_grid =  pool_grid(abs_mins, grid_blocks_flattened, img_size, patch_size)

    # get mx of  max ans abs_min
    abs_maxs = np.maximum(abs_mins, maxs)
    abs_maxs_grid =  pool_grid(abs_maxs, grid_blocks_flattened, img_size, patch_size)

    exp = abs_maxs**2
    exp =  pool_grid(exp, grid_blocks_flattened, img_size, patch_size)

    print(avgs.shape, mins.shape, maxs.shape)
    # make subgrid of hw it al looks
    grids = [img,avgs_grid,mins_grid, maxs_grid, abs_mins_grid, abs_maxs_grid,]
    plot_grids(img, grids, 1, f'different_poolings_{patch_size}')
