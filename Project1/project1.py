import os
import cv2
import sys
import copy
import math
import numpy as np
from tqdm import tqdm


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
NUM_MODELS = 5


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
    """ Returns gradient magnitudes and directions. The direction is given as an angle in degrees. """
    image_width, image_height = image.shape
    grad_x = np.array(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5))
    grad_y = np.array(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5))

    # grad directions represent as (cos\theta, sin\theta)
    grad_mag = np.sqrt(np.square(np.array(grad_x)) + np.square(np.array(grad_y)))
    grad_direction = np.zeros_like(grad_mag)
    for i in range(image_width):
        for j in range(image_height):
            if grad_x[i, j] == 0:
                grad_direction[i, j] = 90
            elif grad_y[i, j] == 0:
                grad_direction[i, j] = 0
            else:
                grad_direction[i, j] = math.atan2(grad_y[i, j], grad_x[i, j]) / math.pi * 180
    return np.array(grad_mag), np.array(grad_direction)


def compute_prob_ang(angle, tau=6, eps=0.1):
    tau = degree_to_radian(tau)
    angle = angle % (2 * math.pi)
    if -tau <= angle <= tau or math.pi - tau <= angle <= math.pi + tau:  # deviation of at most tau
        return (1 - eps) / (4 * tau)
    else:
        return eps / (2 * math.pi - 4 * tau)


def compute_posterior(camera_intrinsics, a, b, g, pixel_locations, pixel_grad_directions):
    a, b, g = degree_to_radian(a), degree_to_radian(b), degree_to_radian(g)
    rot_matrix = helper_functions.angle2matrix(a, b, g)
    posterior = 0

    for i, pixel_location in enumerate(pixel_locations):
        homogenous_location = list(pixel_location) + [1]
        pixel_grad_direction = pixel_grad_directions[i]
        vp_thetas = helper_functions.vp2dir(camera_intrinsics, rot_matrix, homogenous_location)
        for m_idx in range(NUM_MODELS):  # marginalize
            m = m_idx + 1
            if m <= 3:
                vp_theta = vp_thetas[m_idx]
                prob_on = compute_prob_ang(pixel_grad_direction - vp_theta)
                posterior += prob_on
            else:  # m = {4, 5}
                prob_off = 1 / (2 * math.pi)
                posterior += prob_off
    return posterior


def degree_to_radian(angle_in_degrees):
    return angle_in_degrees / 180 * math.pi


def estimate_initial_euler_angles(camera_intrinsics, pixel_locations, pixel_grad_directions):
    """ Reimplementation of https://pdfs.semanticscholar.org/3f12/20be9e783caa716482863af4a671197c6aeb.pdf """
    best_posterior = None
    best_posterior_degrees = [0, 0, 0]

    # search along b first
    print("Searching for best initial euler angles")
    for b in tqdm(range(-45, 46, 4)):
        posterior = compute_posterior(camera_intrinsics, a=0, b=b, g=0, pixel_locations=pixel_locations,
                                      pixel_grad_directions=pixel_grad_directions)
        if best_posterior is None:  # set initial values
            best_posterior = posterior
            best_posterior_degrees[1] = b

        if best_posterior < posterior:
            best_posterior = posterior
            best_posterior_degrees[1] = b

    # medium-scale search:
    medium_search_multipliers = np.array([-1, 0, 1], dtype=float)
    for b in medium_search_multipliers * 2 + best_posterior_degrees[1]:
        for a in medium_search_multipliers * 5:
            for g in medium_search_multipliers * 5:
                posterior = compute_posterior(camera_intrinsics, a=a, b=b, g=g, pixel_locations=pixel_locations,
                                              pixel_grad_directions=pixel_grad_directions)
                if best_posterior < posterior:
                    best_posterior = posterior
                    best_posterior_degrees = [a, b, g]

    # fine-scale-search
    fine_search_multipliers = np.array([-2, -1, 0, 1, 2], dtype=float)
    for a in fine_search_multipliers * 2.5 + best_posterior_degrees[0]:
        for g in fine_search_multipliers * 2.5 + best_posterior_degrees[2]:
            posterior = compute_posterior(camera_intrinsics, a=a, b=best_posterior_degrees[1], g=g,
                                          pixel_locations=pixel_locations, pixel_grad_directions=pixel_grad_directions)
            if best_posterior < posterior:
                best_posterior = posterior
                best_posterior_degrees = [a, best_posterior_degrees[1], g]
    return best_posterior_degrees, best_posterior


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
    initial_euler_angles = estimate_initial_euler_angles(camera_intrinsics, pixel_idxs, pixel_grad_directions)
    print(initial_euler_angles)


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
