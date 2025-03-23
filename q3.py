import numpy as np
import cv2


def shape(points):
    p1, p2, p3, p4 = points
    dists = []
    dists.append(int(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (1 / 2)))
    dists.append(int(((p1[0] - p3[0]) ** 2 + (p1[1] - p3[1]) ** 2) ** (1 / 2)))
    dists.append(int(((p1[0] - p4[0]) ** 2 + (p1[1] - p4[1]) ** 2) ** (1 / 2)))
    dists.remove(max(dists))
    height, width = max(dists), min(dists)
    return height, width


def findHomography(srcPoints):
    heightBook, widthBook = shape(srcPoints)
    dstP1, dstP2, dstP3, dstP4 = [0, 0], [0, widthBook], [heightBook, 0], [heightBook, widthBook]
    dstPoints = np.array([dstP1, dstP2, dstP3, dstP4])
    H, status = cv2.findHomography(srcPoints, dstPoints)
    return H


def warp(img, H, shapeBook):
    heiBook, widBook = shapeBook
    imgWarped = np.zeros((heiBook, widBook, 3))
    iH = np.linalg.inv(H)
    hei, wid, chan = imgWarped.shape
    for ch in range(chan):
        for i in range(hei):
            for j in range(wid):
                loc = [i, j, 1]
                dest = np.matmul(iH, loc)
                dest = dest / dest[2]
                # print(dest)
                x,y = dest[0], dest[1]
                a = x - int(x)
                b = y - int(y)
                A = [1 - a, a]
                B = [1 - b, b]
                x = int(x)
                y = int(y)
                try:
                    F = [[img[x, y, ch], img[x, y + 1, ch]],
                         [img[x + 1, y, ch], img[x + 1, y + 1, ch]]]
                    imgWarped[i, j, ch] = int(np.matmul(np.matmul(A, F), B))
                except:
                    imgWarped[i, j, ch] = 0

    return imgWarped


def findBook(img, p1, p2, p3, p4):
    points = np.array([p1, p2, p3, p4])
    homography = findHomography(points)
    shapeBook = shape(points)
    imgBook = warp(img, homography, shapeBook)
    return imgBook


I = cv2.imread("books.jpg")
Book1 = findBook(I, [210, 665], [395, 600], [105, 380], [290, 320])
Book2 = findBook(I, [740, 350], [710, 155], [465, 405], [425, 205])
Book3 = findBook(I, [975, 800], [1100, 610], [675, 615], [795, 420])
cv2.imwrite("res04.jpg", Book1)
cv2.imwrite("res05.jpg", Book2)
cv2.imwrite("res06.jpg", Book3)
