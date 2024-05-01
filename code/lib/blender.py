import numpy as np
import cv2

class Linear_Blender():
    def __init__(self, blend_ratio=0.1) -> None:
        # percentage of blending window
        self.blend_ratio = blend_ratio

    def blend(self, new_img: np.ndarray, stitch_img: np.ndarray, sx, sy):
        '''
        params:
            sx: int, shift in x direction
            sy: int, shift in y direction
        '''
        new_shifted_imgs = np.roll(new_img, sy, axis=0)

        # Note that new_img is stitched to the left of stitch_img
        new_w = new_img.shape[1]
        half_overlap_x = round((new_w - sx) / 2)
        new_right_bound = new_w - half_overlap_x
        stitch_left_bound = half_overlap_x
        blend_img = np.hstack((new_shifted_imgs[:, :new_right_bound], stitch_img[:, stitch_left_bound:]))

        # Create a linear alpha gradient for blending
        half_blend_x = round(half_overlap_x * self.blend_ratio)
        alpha_gradient = np.linspace(0.5, 0, half_blend_x)
        for j, alpha in enumerate(alpha_gradient):
            blend_img[:, new_right_bound + j] = alpha * new_shifted_imgs[:, new_right_bound + j] + (1 - alpha) * stitch_img[:, stitch_left_bound + j]
            blend_img[:, new_right_bound - j] = (1 - alpha) * new_shifted_imgs[:, new_right_bound - j] + alpha * stitch_img[:, stitch_left_bound - j]

        return blend_img.astype(np.uint8)

  