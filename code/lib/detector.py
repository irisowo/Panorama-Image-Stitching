import numpy as np
import cv2

class Harris_Corner():
    def __init__(self) -> None:
        # params for gaussian kernal
        self.ksize = 3
        self.sigma = 5
        self.k = 0.04 # for response
        self.threshold = 0.03 # local suppression threshold

    def compute_respose(self, img):
        kernal = (self.ksize, self.ksize)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grad_img = cv2.GaussianBlur(gray_img, kernal, self.sigma)
        # dy = cv2.Sobel(gray_img, -1, dx=0, dy=1)
        # dx = cv2.Sobel(gray_img, -1, dx=1, dy=0)

        Iy, Ix = np.gradient(grad_img)
        Ixx, Iyy, Ixy = Ix ** 2, Iy ** 2, Ix * Iy
        Sxx = cv2.GaussianBlur(Ixx, kernal, self.sigma)
        Syy = cv2.GaussianBlur(Iyy, kernal, self.sigma)
        Sxy = cv2.GaussianBlur(Ixy, kernal, self.sigma)
        det_M = Sxx * Syy - Sxy ** 2
        trace_M = Sxx + Syy
        R = det_M - self.k * (trace_M ** 2)
        return R, grad_img


    def detect(self, img):
        # Compute corner response map R
        response, blur_img = self.compute_respose(img)
        response[response <= self.threshold * response.max()] = 0

        # Local suppression
        localmax_kps = np.argwhere((response == cv2.dilate(response, np.ones((3, 3)))) & (response > 0))

        return localmax_kps, blur_img
