import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
VP_THRESHOLD = 1000


def save_vanishing_points(image, homogenous_vanishing_points, filename, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_filepath = os.path.join(save_dir, filename[:-3] + 'png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)

    x, y = list(), list()
    for i in range(3):
        if homogenous_vanishing_points[2, i] == float(0):
            continue  # will be an infinity

        u, v = homogenous_vanishing_points[:2, i] / homogenous_vanishing_points[2, i]
        if -VP_THRESHOLD < u < (IMAGE_WIDTH + VP_THRESHOLD) and -VP_THRESHOLD < v < (IMAGE_HEIGHT + VP_THRESHOLD):
            x.append(u)
            y.append(v)
    plt.scatter(x, y)
    plt.savefig(save_filepath)
    plt.close()


def save_rotation_matrix(rotation_mtx, image_name, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, '.'.join([image_name, 'txt']))
    rotation_mtx = np.array(rotation_mtx, dtype=str)

    with open(filepath, 'w') as f:
        for i in range(len(rotation_mtx)):
            line = rotation_mtx[i]
            line = ' '.join(line) + '\n'
            f.write(line)


def save_assignments(image, assignments, pixel_idxs, image_filename, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_filepath = os.path.join(save_dir, image_filename)
    for i, pixel_idx in enumerate(pixel_idxs):
        assignment = assignments[i]
        assignment_idx = np.argmax(assignment).item()
        if assignment_idx == 3:
            color = (255, 255, 255)
            continue  # we will ignore assignments to other edges
        else:
            color = np.zeros(3)
            color[assignment_idx] = 255
        annotate_pixel(image, *pixel_idx, color=color)
    cv2.imwrite(save_filepath, image)
    return image


def annotate_pixel(image, u, v, color, image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH, region_len=1):
    for i in range(u - region_len, min(u + region_len + 1, image_width - 1)):
        for j in range(v - region_len, min(v + region_len + 1, image_height - 1)):
            image[i, j] = color