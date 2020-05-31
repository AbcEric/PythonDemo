# -*- coding: utf-8 -*-

import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

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

import math
def show_func(X, FUNC):
    for i in range(len(X)):
        x = np.linspace(X[i][0], X[i][1], 100)
        y = eval(FUNC[i])

        plt.figure()  # 定义一个图像窗口
        plt.plot(x, y)  # 绘制曲线 y

    plt.show()


if __name__ == '__main__':
    # print("你好：I♡U")
    # drawPicure("test.jpg", "FUN!")
    #
    # for i in range(1, 100):
    #     prob = simulate(i, loops=100000)
    #     print(i, ": ", prob)


    # 换元要注意：
    # 1.场合：用于值域，不适用于定义域，奇偶性和单调性
    # 2.结构：整体换元，三角换元或均值换元
    # 3.等价：t和x要一一对应，例如若t=3**x(t>0, 改变了x的值域，随着x减小t趋近于0), 若t=x**2（t>0，只是改变方向，没有负值）？

    # show_func([[-10, 10]], ["3**x"])
    # show_func([[-1, 3], [-2, 2], [-5.5, 5.5]], ["x**2 - 2*x + 1", "x**4 - 2*x**2 + 1", "(x**3)**2 - 2*(x**3) + 1"])

    # y=3x+4np.sqrt(1-x**2)  x:[0,1]
    # 若t=np.cos(x) x:[0,3.14/2]
    show_func([[0, 1], [0, 3.14/2]], ["3*x + 4*np.sqrt(1-x**2)", "3*np.cos(x)+4*np.sin(x)"])
