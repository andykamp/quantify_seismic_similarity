# import image module

from PIL import Image

from PIL import ImageFilter

# EDGE ENHANCEMENT FROM https://pythontic.com/image-processing/pillow/edge-enhancement-filter

# Open an already existing image

imageObject = Image.open('/Users/anderskampenes/Documents/Dokumenter/NTNU/MASTER/code/seismic.png')

# Apply edge enhancement filter
edgeEnahnced = imageObject.filter(ImageFilter.EDGE_ENHANCE)

# Apply increased edge enhancement filter
moreEdgeEnahnced = imageObject.filter(ImageFilter.EDGE_ENHANCE_MORE)

# Show original image - before applying edge enhancement filters
#imageObject.show()
imageObject.save('input.bmp')

# Show image - after applying edge enhancement filter
#edgeEnahnced.show()
edgeEnahnced.save('edgeEnahnced.bmp')

# Show image - after applying increased edge enhancement filter
#moreEdgeEnahnced.show()
moreEdgeEnahnced.save('moreEdgeEnahnced.bmp')
