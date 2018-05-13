import numpy as np
from PIL import Image, ImageSequence
import matplotlib.pyplot as plt

# from https://stackoverflow.com/questions/50054187/convert-animated-gif-to-4d-array-in-python
def readGif(path, format='RGBA'):
    channels = 4
    if format != 'RGBA':
        channels = 3

    img = Image.open(path)
    gif = np.array([np.array(frame.copy().convert(format).getdata(),dtype=np.uint8).reshape(frame.size[1],frame.size[0],channels) for frame in ImageSequence.Iterator(img)])
    return gif

if __name__ == "__main__":
    gif = readGif("pokegifs-front/araquanid.gif")
    for i in range(3):
        plt.imshow(gif[i,:,:,:])
        plt.show()
