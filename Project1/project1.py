import os
import cv2
import sys
import copy
import numpy as np


base_dir = os.path.abspath(os.path.dirname(__file__))
if __name__ == '__main__':
    sys.path.append(base_dir)

import EM_help_fucntions as helper_functions

image_filenames = ['P1030001.jpg', 'P1080055.jpg']
image_filepaths = [os.path.join(base_dir, image_filename) for image_filename in image_filenames]
camera_parameters_filepath = os.path.join(base_dir, 'cameraParameters.mat')

result_dir = os.path.join(base_dir, 'results')
grayscale_dir = os.path.join(result_dir, 'grayscale')
pixel_dir = os.path.join(result_dir, 'pixel')

VANISHING_POINT_DIRECTIONS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
EDGE_MODELS_PRIOR = [0.02, 0.02, 0.02, 0.04, 0.09]


def read_image(filename):
    filepath = os.path.join(base_dir, filename)
    return cv2.imread(filepath)


def get_grayscale_image(image):
    """ Converts the image to grayscale and downsamples the image by resizing by a factor of 0.2 """
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image


def get_em_pixel_idxs(grad_mags, grad_directions):
    _, downsampled_idxs = helper_functions.down_sample(Gmag_=grad_mags, Gdir_=grad_directions)
    downsampled_idxs = np.array(downsampled_idxs)
    return downsampled_idxs


def save_image(image, save_dir, save_filename):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, save_filename)
    cv2.imwrite(filepath, image)


def get_image_gradients(image):
    grad_x = np.array(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5))
    grad_y = np.array(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5))

    # grad directions represent as (cos\theta, sin\theta)
    grad_mag = np.sqrt(np.square(np.array(grad_x)) + np.square(np.array(grad_y)))
    grad_direction = np.concatenate([np.expand_dims(grad_x, axis=-1), np.expand_dims(grad_y, axis=-1)], axis=-1)
    return np.array(grad_mag), np.array(grad_direction)


def get_initial_rot_estimate(camera_intrinsics, pixel_grad_mags, pixel_grad_directions):
    """ Reimplementation of https://pdfs.semanticscholar.org/3f12/20be9e783caa716482863af4a671197c6aeb.pdf """
    a, g = 0, 0
    for b in range(-45, 46, 4):
        rot_matrix = helper_functions.angle2matrix(a, b, g)
        vanishing_points = np.matmul(camera_intrinsics, np.matmul(rot_matrix, VANISHING_POINT_DIRECTIONS))
        print(vanishing_points)
        break
    return 1


def annotate_pixel_locations(image, pixel_locations, save_filename, region_size=1, color=(0, 0, 255)):
    image_to_annotate = copy.deepcopy(image)
    locations_to_annotate = copy.deepcopy(pixel_locations) * 5 + 4
    image_width, image_height, _ = image.shape
    for u, v in locations_to_annotate:
        for i in range(u-region_size, min(u+region_size+1, image_width-1)):
            for j in range(v-region_size, min(v+region_size+1, image_height-1)):
                image_to_annotate[i, j] = color
    save_image(image_to_annotate, save_dir=pixel_dir, save_filename=save_filename)


def process_image(image_filename):
    image = read_image(image_filename)
    camera_intrinsics = helper_functions.cam_intrinsics(camera_parameters_filepath)

    # Step 1: preprocess.
    grayscale_image = get_grayscale_image(image)
    save_image(grayscale_image, save_dir=grayscale_dir, save_filename=image_filename)
    grad_mags, grad_directions = get_image_gradients(grayscale_image)
    pixel_idxs = get_em_pixel_idxs(grad_mags=grad_mags, grad_directions=grad_directions)
    annotate_pixel_locations(image, pixel_idxs, image_filename)

    pixel_grad_mags, pixel_grad_directions = get_pixel_gradients(grad_mags, grad_directions, pixel_idxs)
    print(pixel_grad_mags.shape)
    print(pixel_grad_directions.shape)
    rot_estimate = get_initial_rot_estimate(camera_intrinsics, pixel_grad_mags, pixel_grad_directions)


def get_pixel_gradients(grad_mags, grad_directions, pixel_idxs):
    downsampled_grad_directions = grad_directions[4::5]
    downsampled_grad_mags = grad_mags[4::5]
    pixel_grad_mags, pixel_grad_directions = list(), list()
    for i, j in pixel_idxs:
        pixel_grad_mags.append(downsampled_grad_mags[i, j])
        pixel_grad_directions.append(downsampled_grad_directions[i, j])
    return np.array(pixel_grad_mags), np.array(pixel_grad_directions)


if __name__ == '__main__':
    for image_filename in image_filenames:
        process_image(image_filename)
