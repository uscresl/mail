import os
import cv2
import numpy as np
import imageio
import glob
from PIL import Image
from moviepy.editor import ImageSequenceClip
import cv2


def make_grid(array, nrow=1, padding=0, pad_value=120):
    """ numpy version of the make_grid function in torch. Dimension of array: NHWC """
    if len(array.shape) == 3:  # In case there is only one channel
        array = np.expand_dims(array, 3)
    N, H, W, C = array.shape
    assert N % nrow == 0
    ncol = N // nrow
    idx = 0
    grid_img = None
    for i in range(nrow):
        row = np.pad(array[idx], [[padding if i == 0 else 0, padding], [padding, padding], [0, 0]], constant_values=pad_value)
        for j in range(1, ncol):
            idx += 1
            cur_img = np.pad(array[idx], [[padding if i == 0 else 0, padding], [0, padding], [0, 0]], constant_values=pad_value)
            row = np.hstack([row, cur_img])
        if i == 0:
            grid_img = row
        else:
            grid_img = np.vstack([grid_img, row])
    return grid_img


if __name__ == '__main__':
    N = 12
    H = W = 50
    X = np.random.randint(0, 255, size=N * H * W* 3).reshape([N, H, W, 3])
    grid_img = make_grid(X, nrow=3, padding=5)
    cv2.imshow('name', grid_img / 255.)
    cv2.waitKey()


def save_numpy_as_gif(array, filename, fps=20, scale=1.0):
    """Creates a gif given a stack of images using moviepy
    Notes
    -----
    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
    Usage
    -----
    >>> X = randn(100, 64, 64)
    >>> gif('test.gif', X)
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """

    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.gif'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_gif(filename, fps=fps)
    return clip


def save_numpy_to_gif_matplotlib(array, filename, interval=50):
    from matplotlib import animation
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    def img_show(i):
        plt.imshow(array[i])
        print("showing image {}".format(i))
        return

    ani = animation.FuncAnimation(fig, img_show, len(array), interval=interval)

    ani.save('{}.mp4'.format(filename))

    import ffmpy
    ff = ffmpy.FFmpeg(
        inputs={"{}.mp4".format(filename): None},
        outputs={"{}.gif".format(filename): None})

    ff.run()
    # plt.show()

def save_numpy_to_mp4(array, filename, interval=50):
    width = array.shape[1]
    hieght = array.shape[2]
    fps = 20
    
    # handle backward compatibility: replace .gif filename to .mp4
    filename = filename.replace('gif', 'mp4')

    # RGB to BGR (cv2 format): https://www.scivision.dev/numpy-image-bgr-to-rgb/
    array = array[...,::-1].copy()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # FourCC is a 4-byte code used to specify the video codec.
    # A video codec is software or hardware that compresses and decompresses digital video. 
    # In the context of video compression, codec is a portmanteau of encoder and decoder, 
    # while a device that only compresses is typically called an encoder, and one that only 
    # decompresses is a decoder. Source - Wikipedia
    
    #Syntax: cv2.VideoWriter( filename, fourcc, fps, frameSize )
    video = cv2.VideoWriter(filename, fourcc, float(fps), (width, hieght))
    
    for image in array:
        video.write(image)

    video.release()
    print(f'{filename} saved!')
