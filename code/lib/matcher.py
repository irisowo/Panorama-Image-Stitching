from scipy.spatial import distance
import numpy as np


class Brute_Force_Matcher():
    def __init__(self) -> None:
        self.ratio = 0.75

    def match(self, desc1, desc2):
        matches = []
        #  Brute force : compute distance matrix of all pairs
        dist_matrix = distance.cdist(desc1, desc2)
        
        for i in range(dist_matrix.shape[0]):
            dists = dist_matrix[i]
            best_idx, second_idx = np.argsort(dists)[:2]
            # Ratio Test
            if (dists[best_idx] / dists[second_idx]) < self.ratio:
                matches.append([i, best_idx])

        return np.array(matches)