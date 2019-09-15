import cv2
import numpy as np

class SeamCarver:
    def __init__(self, filename, out_height, out_width):
        self.out_height = out_height
        self.out_width = out_width

        # read in image
        self.in_image = cv2.imread(filename).astype(np.float64)
        self.in_height, self.in_width = self.in_image.shape[: 2]

        # keep tracking resulting image
        self.out_image = np.copy(self.in_image)
        # some large number
        self.constant = 10000000

        # start
        self.seams_carving()


    def seams_carving(self):
        delta_row, delta_col = int(self.out_height - self.in_height), int(self.out_width - self.in_width)

        if delta_col < 0:
            self.seams_removal(delta_col * -1)


    def seams_removal(self, num_pixel):
        for dummy in range(num_pixel):
            energy_map = self.calc_energy_map()
            seam_idx = self.viterbi(energy_map)

            self.delete_seam(seam_idx)


    def calc_energy_map(self):
        b, g, r = cv2.split(self.out_image)
        b_energy = np.absolute(cv2.Scharr(b, -1, 1, 0)) + np.absolute(cv2.Scharr(b, -1, 0, 1))
        g_energy = np.absolute(cv2.Scharr(g, -1, 1, 0)) + np.absolute(cv2.Scharr(g, -1, 0, 1))
        r_energy = np.absolute(cv2.Scharr(r, -1, 1, 0)) + np.absolute(cv2.Scharr(r, -1, 0, 1))
        return b_energy + g_energy + r_energy



    def delete_seam(self, seam_idx):
        m, n = self.out_image.shape[: 2]
        output = np.zeros((m, n - 1, 3))
        for row in range(m):
            col = seam_idx[row]
            output[row, :, 0] = np.delete(self.out_image[row, :, 0], [col])
            output[row, :, 1] = np.delete(self.out_image[row, :, 1], [col])
            output[row, :, 2] = np.delete(self.out_image[row, :, 2], [col])
        self.out_image = np.copy(output)

    def save_result(self, filename):
        cv2.imwrite(filename, self.out_image.astype(np.uint8))

    def viterbi(self,energy_map):
        #TODO
        pass




if __name__ == '__main__':
    filename_input = 'image_input.jpg'
    filename_output = 'image_output.jpg'


    height_input, width_input = cv2.imread(filename_input).astype(np.float64).shape[: 2]

    height_output = height_input
    width_output = width_input - 30
    print('Original image size: ', height_input,width_input)

    obj = SeamCarver(filename_input, height_output, width_output)
    obj.save_result(filename_output)







