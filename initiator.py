import cv2 as cv
import numpy as np

def pixelmatrix(matrix, pixel=255):
    """
    input image and get image. shape +256 shape of output matrix with every matrix of pixel wise split in image array
    """
    shape = list(matrix.shape)
    shape.append(pixel)
    for x in range(2):
        shape[x] = int(np.ceil(shape[x] / 10) * 10)
    output = np.zeros(shape)
    for z in range(shape[2]):
        for p in range(1, pixel + 1):
            xp, yp = np.where(matrix[:, :, z] == p)
            output[xp, yp, z, p - 1] = 1
    return output

def anchorcreater(matrix, anchorshape=[10, 10]):
    """
    input pixel-image shape like img.shape+256 and get output of [10,10,x/10,y/10,*_,256]
    """
    shape = anchorshape + list(matrix.shape)
    x, y = [int(z / 10) for z in shape[2:4]]
    for index in range(2, 4):
        shape[index] = int(shape[index] / 10)
    output = np.zeros(shape)
    ymin, ymax = 0, 10
    for yanchor in range(y):
        xmin = 0
        xmax = 10
        for xanchor in range(x):
            output[:, :, xanchor, yanchor, :, :] = matrix[xmin:xmax, ymin:ymax, :, :]
            xmin += 10
            xmax += 10
        ymin += 10
        ymax += 10
    return output


def last_matrix(image):
    return anchorcreater(pixelmatrix(image))

if __name__ == '__main__':
    img = cv.imread("picture.jpg")
    output = last_matrix(img)
    print(output.shape)

