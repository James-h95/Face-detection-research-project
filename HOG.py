import cv2
import numpy as np



class HOG:
    def __init__(self, cell_size=8, block_size=2, bins=9):
        self.cell_size = cell_size
        self.block_size = block_size
        self.bins = bins
        self.kernelx = np.array([[0,0,0],[-1,0,1],[0,0,0]])
        self.kernely = np.array([[0,-1,0],[0,0,0],[0,1,0]])
        self.angle_unit = 180 / self.bins

    def load_image_as_gray_matrix(self, image_path):
        self.img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        self.img = self.img.astype(np.float32) 
        return self.img

    def histogram(self, window):
        height, width = window.shape
        numCellX = width // self.cell_size
        numCellY = height // self.cell_size
        grad_x = cv2.filter2D(self.img, -1, self.kernelx)
        grad_y = cv2.filter2D(self.img, -1, self.kernely)
        rad = np.arctan2(grad_y, grad_x)
        angle_deg = np.degrees(rad)
        angle_deg = np.where(angle_deg < 0, angle_deg + 180, angle_deg)
        hist = np.zeros((numCellX, numCellY, self.bins))
        features = np.zeros(((numCellX- self.block_size + 1)*(numCellY - self.block_size + 1), self.bins * self.block_size * self.block_size))
        for i in range(0, numCellX):
            for j in range(0, numCellY):
                for x in range(i * self.cell_size, i * self.cell_size + self.cell_size):
                    for y in range(j * self.cell_size, j * self.cell_size + self.cell_size):
                        index  =  int(np.floor(angle_deg[y][x] / self.angle_unit))
                        if index == self.bins: 
                            index -= 1
                        hist[i][j][index] += 1
        numFeatureX = numCellX- self.block_size + 1
        numFeatureY = numCellY - self.block_size + 1
        for i in range(numFeatureX):
            for j in range(numFeatureY):
                start = 0 
                for cx in range(0, self.block_size):
                    for cy in range(0, self.block_size):
                        features[i + j * numFeatureX][start : start + self.bins  ] =hist[i + cx][j+cy]   
                        start += self.bins   
                norm = np.linalg.norm(features[i + j * numFeatureX])
                features[ i + j * numFeatureX] /= norm

        return features
  

