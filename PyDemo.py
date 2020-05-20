# -*- coding: utf-8 -*-

import random
import numpy as np
import cv2

BLACK = 1       # BLACK first, if the last turn is BLACK, BLACK win!
WHITE = 0

def simulate(step, loops=1000):
    win = 0
    turn = BLACK

    for i in range(loops):
        for j in range(step):
            if j == step-1:
                if turn == BLACK:
                    win += 1
                turn = BLACK
            else:
                num = random.random()
                if num >= 1/5:          # if num<0.333, continue take. otherwise the oppenent takes!
                    turn = 1-turn

    return win/loops


def drawPicure(photo, draw_text):
    img = cv2.imread(photo)
    H = img.shape[0]
    W = img.shape[1]
    blank = np.zeros((2*H, 2*W, 3), dtype=np.uint8)
    blank[:, :, :] = 255
    # m = 9
    n = 4

    for i in range(0, H, n):
        # rand = random.randint(0, len(draw_text)-1)
        rand = 0
        for j in range(0, W, n):
            # print(i, j, draw_text[(int(j / n) + rand) % len(draw_text)])
            cv2.putText(blank, draw_text[int(j/n+rand) % len(draw_text)], (2*j, 2*i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (int(img[i][j][2]), int(img[i][j][1]), int(img[i][j][0])), 1)

    cv2.imshow('img', blank)
    cv2.imwrite('beau.jpg', blank)

    cv2.waitKey(0)


if __name__ == '__main__':
    print("你好：I♡U")
    drawPicure("test.jpg", "ILOVEU")

    exit(0)

    for i in range(1, 100):
        prob = simulate(i, loops=100000)
        print(i, ": ", prob)