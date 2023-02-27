import cv2
import numpy as np

k1 = 50
k2 = 0

img = cv2.imread("english-black-lab-puppy.jpg", cv2.IMREAD_GRAYSCALE)

length, width = img.shape
midpoint = np.array([length // 2, width // 2], dtype=float)

def distance(point):
    return np.sqrt(np.dot(point, point))

def distorted_point(point):
    from_midpoint = point - midpoint
    dist = distance(from_midpoint)
    lam = 1 + k1 * (1 / dist**2) + k2 * (dist ** 4)

    return midpoint + (from_midpoint // lam)

buffer = img.copy()

for j in range(length):
    for i in range(width):
        point = np.array([j, i], dtype=int)
        distorted = distorted_point(point)
        if distorted[0] < 0 or distorted[1] < 0 or distorted[0] >= length or distorted[1] >= width:
            buffer[j, i] = 0
        else:
            buffer[j, i] = img[int(distorted[0]), int(distorted[1])]

img = buffer

cv2.imshow("asd", img)

cv2.waitKey()
