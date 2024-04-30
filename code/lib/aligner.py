import numpy as np
import random


class RANSAC_Aliner():
    def __init__(self) -> None:
        self.ransac_iter = 500
        self.threshold  = 8

    def align (self, kp1:np.ndarray, kp2: np.ndarray):
        shifts = kp1 - kp2

        best_shifts = np.zeros(2)
        best_inlier_indices = []
        for i in range(self.ransac_iter):
            random.seed(i)
            rand_shifts = np.mean(random.choices(shifts, k=6), axis=0)
            
            # np.where return a tuple of array t, where t[0] is the row index and t[1] is the column index
            inlier_indices = np.where(abs(shifts - rand_shifts) < self.threshold)[0]
    
            # update if the random points give the largest number of inliers
            if len(inlier_indices) > len(best_inlier_indices):
                best_inlier_indices = inlier_indices
                # count twice if both x, y < threshold, count once if only one of them < threshold
                best_shifts = np.mean(shifts[inlier_indices], axis = 0)

        shifts_y, shifts_x = best_shifts[0], best_shifts[1]
        return abs(round(shifts_x)), round(shifts_y)