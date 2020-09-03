import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from skimage import io
from progress.bar import Bar


def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('input_image', help='Path of image to be compared.')
    parser.add_argument('search_path', help='Path to search for images to compare.')
    parser.add_argument('--number_of_matches', default=10, type=int, help='Number of matches to be returned. Default = 10')
    parser.add_argument('--image_extensions', nargs='+', default=['jpg', 'png'], type=str, help='Image extensions allowed. Default = [jpg, png]. Type new image extensions to add.')
    parser.add_argument('--save', default=None, type=str, help='Saves the graphics and results if a path is entered.')

    args = parser.parse_args()
    return args


def plot(image1, image2, histogram1, histogram2, image1_name, image2_name, distance, save):
    """
    Plots or saves images and histograms.

    :param image1: Searched image.
    :param image2: Found image.
    :param histogram1: Histogram of searched image.
    :param histogram2: Histogram of found image.
    :param image1_name: Name of input_image.
    :param image2_name: Name of found image.
    :param distance: Distance between image1 and image2.
    :param save: Path to save results.
    """

    channels = histogram1.shape[0]
    range_256 = np.arange(256)
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(19, 10))
    plt.suptitle('Distance: ' + str(distance))
    ax1.set_title(image1_name)
    ax2.set_title(image2_name)

    if channels == 1:
        ax1.imshow(image1, cmap='gray')
        ax2.imshow(image2, cmap='gray')
        ax3.bar(range_256, histogram1[0], color='black')
        ax4.bar(range_256, histogram2[0], color='black')
    elif channels == 3:
        ax1.imshow(image1)
        ax2.imshow(image2)
        colors = ('red', 'green', 'blue')
        for channel, color in zip(range(3), colors):
            ax3.bar(range_256, histogram1[channel], color=color, alpha=0.5)
            ax4.bar(range_256, histogram2[channel], color=color, alpha=0.5)
    else:
        ax1.imshow(image1, cmap='gray')
        ax2.imshow(image2, cmap='gray')
        for channel in channels:
            ax3.bar(range_256, channel, alpha=0.5)
            ax4.bar(range_256, channel, alpha=0.5)

    if save:
        if not os.path.exists(save):
            os.makedirs(save)
        plt.savefig(os.path.join(save, image2_name))
    plt.show()
    plt.close()


def list_image_files(path, extensions, images_list):
    """
    Recursively search for images in a directory and save them to a list.

    :param path: Directory for searching images.
    :param extensions: Image extensions allowed.
    :param images_list: List to save allowed images.
    """

    files = os.listdir(path)

    for file in files:
        file_path = os.path.join(path, file)
        absolute_path = os.path.abspath(file_path)
        if os.path.isdir(absolute_path):
            list_image_files(absolute_path, extensions, images_list)
        else:
            if file.endswith(extensions):
                images_list.append(absolute_path)


def get_dtype(size):
    """
    Since an image can contain only black or white pixels, the size of the integer to be used in the histogram is configured according to the height and width of the image.

    :param size: Image size (number of pixels).
    :return: An unsigned numpy integer type.
    """

    if size < 256:
        return np.uint8
    if size < 65536:
        return np.uint16
    if size < 4294967295:
        return np.uint32
    return np.uint64


def create_gray_histogram(image):
    """
    Creates a numpy array with dimension (1, 256) to store the histogram of a grayscale image. Each array index (0, 1, 2, ..., 255) represents a pixel value and the value of each position in the array represents the frequency of each pixel in the image.

    :param image: Gray scale image.
    :return: Numpy array with the frequency of the pixel values in a gray scale image.
    """

    height, width = image.shape
    size = height * width
    dtype = get_dtype(size)
    histogram = np.zeros(shape=(1, 256), dtype=dtype)

    flatten_image = image.flatten()
    for index in flatten_image:
        histogram[0, index] += 1
    return histogram


def create_rgb_histogram(image):
    """
    Creates a numpy array with dimension (n, 256) (n = number of channels of image) to store the histogram of a multi-channel image. Each array index ([0, 0], [1, 0], [2, 0], ..., [255, 0]; [0, 1], [1, 1], [2, 1] , ..., [255, 1]; [0, n-1], [1, n-1], [2, n-1], ..., [255, n-1]) represents a value pixel for a given color channel and the value of each position of the array represents the frequency of each pixel in the image in its respective channel.

    :param image: Multi-channel image.
    :return: Numpy array with the frequency of the pixel values in a multi-channel image.
    """

    height, width, channels = image.shape
    size = height * width
    dtype = get_dtype(size)
    histogram = np.zeros(shape=(channels, 256), dtype=dtype)

    for h in range(height):
        for w in range(width):
            for c in range(channels):
                index = image[h, w, c]
                histogram[c, index] += 1
    return histogram


def calculate_histogram(image):
    """
    Load the histogram of an image. Since an image can contain more than one channel, the number of channels in the histogram is configured according to the number of channels in the image.

    :param image Loaded image.
    :return: a numpy array with the image histogram.
    """

    channels = 1
    if len(image.shape) != 2:
        channels = image.shape[2]

    if channels == 1:
        histogram = create_gray_histogram(image)
    else:
        histogram = create_rgb_histogram(image)
    return histogram


def calculate_pdf(histogram, image_size):
    """
    Compute the Probability Density Function (PDF) of a histogram with image_size (number of pixels) positions. The value of each position is divided by image_size.

    :param histogram: Histogram of an image.
    :param image_size: Number of pixels of an image. Can be obtained by multiplying height x width.
    :return: A numpy array of size image_size with each histogram value divided by number of pixels.
    """

    histogram = histogram / image_size
    return histogram


def calculate_distance(pdf1, pdf2, channels):
    """
    Calculates the Euclidean Distance between two PDFs.

    :param pdf1: PDF of input image.
    :param pdf2: PDF of the image to be compared.
    :param channels: Number of channels.
    :return: A float number that represents the calculated distance.
    """

    distances = np.zeros(shape=(1, 256*channels))
    pdf1_flatten = pdf1.flatten()
    pdf2_flatten = pdf2.flatten()

    index = 0
    for p1, p2 in zip(pdf1_flatten, pdf2_flatten):
        distances[0, index] = (p1 - p2) ** 2
        index += 1
    return np.sum(distances)


def multichannel2gray(image):
    """
    Convert multi-channel image to a gray scale image using the pixel average according to the number of channels.

    :param image: A multi-channel image.
    :return: A gray scale image.
    """

    height, width, channels = image.shape
    gray_image = np.zeros(shape=(height, width), dtype=np.uint8)

    for h in range(height):
        for w in range(width):
            gray_image[h, w] = int(np.sum(image[h, w]) / channels)
    return gray_image


def search_matches(input_image_path, images_list, number_of_matches, save):
    """
    Given an input image and a set of images, it searches for images similar to the input image.

    :param input_image_path: Path to input image.
    :param images_list: Path to set of images. Allows subdirectories.
    :param number_of_matches: N images with shorter distances from the input image.
    :param save: Path to save results.
    """

    input_image_name = input_image_path.split('/')[-1]
    input_image = io.imread(input_image_path)
    input_image_histogram = calculate_histogram(input_image)
    input_image_channels = input_image_histogram.shape[0]
    input_image_size = input_image.shape[0] * input_image.shape[1]
    input_image_pdf = calculate_pdf(input_image_histogram, input_image_size)

    bar = Bar('Loading histograms, calculating PDF and measuring distances...', max=(len(images_list)))
    distances = dict()
    for image_path in images_list:
        image = io.imread(image_path)
        image_histogram = calculate_histogram(image)
        image_channels = image_histogram.shape[0]
        image_size = image.shape[0] * image.shape[1]
        image_pdf = calculate_pdf(image_histogram, image_size)

        # If both images have the same number of channels then the distance is calculated pixel by pixel with the corresponding channel.
        if input_image_channels == image_channels:
            distance = calculate_distance(input_image_pdf, image_pdf, input_image_channels)

        # If the images differ in the number of channels, they are all converted to gray scale
        else:
            image1 = input_image.copy()
            channels1 = input_image_channels
            size1 = input_image_size
            pdf1 = input_image_pdf.copy()

            # Convert input_image to grayscale and recalculate histogram and pdf
            if channels1 != 1:
                image1 = multichannel2gray(image1)
                histogram1 = calculate_histogram(image1)
                pdf1 = calculate_pdf(histogram1, size1)

            image2 = image.copy()
            channels2 = image_channels
            size2 = image_size
            pdf2 = image_pdf.copy()

            # Convert image to be compared to grayscale and recalculate histogram and pdf
            if channels2 != 1:
                image2 = multichannel2gray(image2)
                histogram2 = calculate_histogram(image2)
                pdf2 = calculate_pdf(histogram2, size2)

            distance = calculate_distance(pdf1, pdf2, 1)

        distances[image_path] = distance
        bar.next()
    bar.finish()

    df = pd.DataFrame(data=distances.values(), columns=['distance'], index=distances.keys())
    df = df.sort_values(by='distance')

    index = df.index
    for i in index[:number_of_matches]:
        image_name = i.split('/')[-1]
        image = io.imread(i)
        image_histogram = calculate_histogram(image)
        image_channels = image_histogram.shape[0]

        distance = df[df.index == i]['distance'][-1]
        print(input_image_name, '<--->', image_name, '=', distance)

        if input_image_channels == image_channels:
            plot(input_image, image, input_image_histogram, image_histogram, input_image_name, image_name, distance, save)
        else:
            image1 = input_image.copy()
            channels1 = input_image_channels
            histogram1 = input_image_histogram.copy()

            if channels1 != 1:
                image1 = multichannel2gray(image1)
                histogram1 = calculate_histogram(image1)

            image2 = image.copy()
            channels2 = image_channels
            histogram2 = image_histogram.copy()

            if channels2 != 1:
                image2 = multichannel2gray(image2)
                histogram2 = calculate_histogram(image2)

            plot(image1, image2, histogram1, histogram2, input_image_name, image_name, distance, save)


def main():
    args = arg_parse()

    extensions = ['jpg', 'png']
    extensions += args.image_extensions

    images_list = list()  # list to save the path of the images
    list_image_files(args.search_path, tuple(extensions), images_list)
    search_matches(args.input_image, images_list, args.number_of_matches, args.save)


if __name__ == '__main__':
    main()
