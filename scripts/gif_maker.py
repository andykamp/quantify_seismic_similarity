import matplotlib
matplotlib.use(
    "TkAgg"
)  # suggested to avoid AttributeError: 'FigureCanvasMac' object has no attribute 'renderer'
import matplotlib.pyplot as plt
import numpy as np
import imageio


def plot_slice(slice):
    fig = plt.figure()
    # remove plots and margins
    plt.axis('off')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.imshow(slice)
    #plt.savefig('seismic.png')

    # IMPORTANT ANIMATION CODE HERE
    # Used to keep the limits constant
    # ax.set_ylim(0, y_max)

    # Used to return the plot as an image rray
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image

# get data to use 
data_cube = np.load("/Users/anderskampenes/Documents/Dokumenter/NTNU/MASTER/code/data/raw/f3-benchmark/test_once/test1_seismic.npy")
# make sure it is fomrated correctly 
data_cube = data_cube[:,:200,:].T
print("Shapes: ", data_cube.shape)
SHAPE_INLINE = data_cube.shape[0 ]
SHAPE_XLINE = data_cube.shape[1 ]
SHAPE_DEPTH = data_cube.shape[2 ]
FPS = 10
#
#kwargs_write = {"fps": FPS, "quantizer": "nq"}
#imageio.mimsave(
#    "./np_to_gif.gif",
#    [
#        plot_slice(data_cube[:,  i,  :].reshape(SHAPE_INLINE, SHAPE_DEPTH))
#        for i in range(SHAPE_XLINE - 1 if SHAPE_XLINE > 1 else SHAPE_XLINE)
#    ],
#    fps=FPS,
#)


writer = imageio.get_writer('test.mp4', fps=FPS)

for i in range(SHAPE_XLINE - 1 if SHAPE_XLINE > 1 else SHAPE_XLINE):
    img = plot_slice(data_cube[ :, i, :].reshape(SHAPE_INLINE, SHAPE_DEPTH))
    writer.append_data(img)
writer.close()