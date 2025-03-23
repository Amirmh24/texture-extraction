import numpy as np
import cv2


class Box:
    def __init__(self, i1, j1, i2, j2):
        self.i1, self.j1, self.i2, self.j2 = i1, j1, i2, j2
        self.hei = i2 - i1
        self.wid = j2 - j1

    def resize(self, a):
        self.i1, self.j1, self.i2, self.j2 = self.i1 * a, self.j1 * a, self.i2 * a, self.j2 * a
        return self


def drawBoxes(img, boxes):
    for r in range(len(boxes)):
        box = boxes[r]
        img = cv2.rectangle(img, (box.j1, box.i1), (box.j2, box.i2), (0, 0, 255), 2)
    return img


def matchTemplate(img, patch, boxes):
    height, width, channels = img.shape
    heiP, widP, chanP = patch.shape
    g = patch
    gm = np.mean(g)
    for i in range(height - heiP):
        print(i)
        for j in range(width - widP):
            f = img[i:i + heiP, j:j + widP, :]
            fm = np.mean(f)
            corr = (np.sum((f - fm) * (g - gm))) / (np.sum((f - fm) ** 2) * np.sum((g - gm) ** 2)) ** (1 / 2)
            if (corr > 0.78):
                boxes.append(Box(i, j, i + heiP, j + widP))
    return boxes


def removeExtraBoxes(boxes):
    optimalBoxes = []
    i = 0
    while i < len(boxes) - 1:
        j = 0
        ui1, uj1, ui2, uj2 = boxes[i].i1, boxes[i].j1, boxes[i].i2, boxes[i].j2
        while j < len(boxes):
            if ((i != j) & (((boxes[i].j1 == boxes[j].j1)&(boxes[i].j2 == boxes[j].j2)) |
                            ((boxes[i].i1 < boxes[j].i1) & (boxes[i].j1 < boxes[j].j1) & (boxes[i].i2 > boxes[j].i2) & (
                                    boxes[i].j2 > boxes[j].j2)))):
                ui1, uj1 = min(boxes[i].i1, boxes[j].i1), min(boxes[i].j1, boxes[j].j1)
                ui2, uj2 = max(boxes[i].i2, boxes[j].i2), max(boxes[i].j2, boxes[j].j2)
                boxes.pop(j)
                j = 0
            j = j + 1
        optimalBoxes.append(Box(ui1, uj1, ui2, uj2))
        i = i + 1
    return optimalBoxes


a = 8
I = cv2.imread("Greek_ship.jpg").astype(np.float64)
patch = cv2.imread("patch.png").astype(np.float64)
height, width, channels = I.shape
patch = patch[50:, 60:140, :]
heiP, widP, chanP = patch.shape
I = cv2.resize(I, (int(width / a), int(height / a)))
patch = cv2.resize(patch, (int(widP / a), int(heiP / a)))
height, width, channels = I.shape
heiP, widP, chanP = patch.shape

h = 7 / 8
rr = 1
boxes = []
for i in range(4):
    patch = cv2.resize(patch, (int(widP * rr), int(heiP * rr)))
    heiP, widP, chanP = patch.shape
    boxes = matchTemplate(I, patch, boxes)
    rr = rr * h
boxes=removeExtraBoxes(boxes)
I = cv2.imread("Greek_ship.jpg")
for i in range(len(boxes)):
    boxes[i] = boxes[i].resize(a)
I = drawBoxes(I, boxes)
cv2.imwrite("res03.jpg", I)
