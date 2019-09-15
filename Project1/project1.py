import os
import cv2
import sys
import math
import numpy as np


base_dir = os.path.abspath(os.path.dirname(__file__))
if __name__ == '__main__':
    sys.path.append(base_dir)

import EM_help_fucntions as helper_functions

image_filenames = ['P1030001.jpg', 'P1080055.jpg']
image_filepaths = [os.path.join(base_dir, image_filename) for image_filename in image_filenames]
camera_parameters_filepath = os.path.join(base_dir, 'cameraParameters.mat')

result_dir = os.path.join(base_dir, 'results')
preprocess_dir = os.path.join(result_dir, 'gray_scale')


def read_images(filepaths):
    return [cv2.imread(filepath) for filepath in filepaths]


def get_grayscale_images(images):
    """ Converts the image to grayscale and downsamples the image by resizing by a factor of 0.2 """
    grayscale_images = list()
    for image in images:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayscale_images.append(gray_image)
    return grayscale_images


def get_em_pixel_idxs(grad_mags, grad_directions):
    downsampled_idxs = list()
    for i, grad_mag in enumerate(grad_mags):
        grad_direction = grad_directions[i]
        _, downsampled_idx = helper_functions.down_sample(Gmag_=grad_mag, Gdir_=grad_direction)
        downsampled_idxs.append(downsampled_idx)
    downsampled_idxs = np.array(downsampled_idxs)
    return downsampled_idxs


def save_image(image, filepath):
    save_dir = os.path.dirname(filepath)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cv2.imwrite(filepath, image)


def save_images(images, save_filenames, save_dir):
    assert len(images) == len(save_filenames)
    for i, image in enumerate(images):
        save_filepath = os.path.join(save_dir, save_filenames[i])
        save_image(image, save_filepath)


def get_grad_direction(x_grads, y_grads):
    directions = np.zeros_like(x_grads)
    assert x_grads.shape == y_grads.shape
    r, c = x_grads.shape
    for i in range(r):
        for j in range(c):
            x_grad = x_grads[i, j]
            y_grad = y_grads[i, j]
            if x_grad == 0:  # implies that the direction is along the y-axis
                directions[i, j] = math.pi / 2
            elif y_grad == 0:  # implies that it is along the x-axis
                directions[i, j] = 0
            else:  # normal scenario
                directions[i, j] = math.atan(y_grad / x_grad)
    return directions


def get_gradients(images):
    grad_mags, grad_directions = list(), list()
    for image in images:
        grad_x = np.array(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5))
        grad_y = np.array(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5))
        grad_mag = np.sqrt(np.square(np.array(grad_x)) + np.square(np.array(grad_y)))
        grad_direction = get_grad_direction(grad_x, grad_y)
        grad_mags.append(grad_mag)
        grad_directions.append(grad_direction)
    return np.array(grad_mags), np.array(grad_directions)


def main():
    images = read_images(image_filepaths)
    camera_params = helper_functions.cam_intrinsics(camera_parameters_filepath)

    # Step 1: preprocess.
    grayscale_images = get_grayscale_images(images)
    save_images(grayscale_images, image_filenames, save_dir=preprocess_dir)
    grad_mags, grad_directions = get_gradients(grayscale_images)

    em_pixel_idxs = get_em_pixel_idxs(grad_mags=grad_mags, grad_directions=grad_directions)

    print(camera_params)


if __name__ == '__main__':
    main()
