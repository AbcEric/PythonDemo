# !/usr/bin/env python3

"""
File: CircleOrbit.py

This Python 3 code is designed to generate the inner_circle orbit, get image file from video frame by frame.

Author:     Eric Lee
Date:	    4/30/2020
Version     1.0.0
License:    ABC

History:    1. Try slow_type video file;
            2. Sometimes the yellow mask may be hidden;
            3. Determine the yellow mask position automatical by opencv;
            4. Animation: change from turtle to opencv（0504）;
            5. Equally sample: should according to speed：or judge whethether display or not when showorbits()
            6. Smooth the curve and record to video (0505);
"""

import cv2
import os
import numpy as np
from math import factorial, sqrt
import matplotlib.pyplot as plt
import pkg_resources.py2_warn

#
# Bezier曲线拟合：粗略
#
def evaluate_bezier(points, total):

    def comb(n, k):
        return factorial(n) // (factorial(k) * factorial(n - k))

    def get_bezier_curve(points):
        n = len(points) - 1
        return lambda t: sum(
            comb(n, i) * t ** i * (1 - t) ** (n - i) * points[i]
            for i in range(n + 1)
        )

    bezier = get_bezier_curve(points)
    new_points = np.array([bezier(t) for t in np.linspace(0, 1, total)])
    return new_points[:, 0], new_points[:, 1]


def bezier_curve(p0, p1, p2, p3, inserted):
    """
    三阶贝塞尔曲线

    p0, p1, p2, p3 - 点坐标，tuple、list或numpy.ndarray类型
    inserted  - p0和p3之间插值的数量
    """

    assert isinstance(p0, (tuple, list, np.ndarray)), u'点坐标不是期望的元组、列表或numpy数组类型'
    assert isinstance(p0, (tuple, list, np.ndarray)), u'点坐标不是期望的元组、列表或numpy数组类型'
    assert isinstance(p0, (tuple, list, np.ndarray)), u'点坐标不是期望的元组、列表或numpy数组类型'
    assert isinstance(p0, (tuple, list, np.ndarray)), u'点坐标不是期望的元组、列表或numpy数组类型'

    if isinstance(p0, (tuple, list)):
        p0 = np.array(p0)
    if isinstance(p1, (tuple, list)):
        p1 = np.array(p1)
    if isinstance(p2, (tuple, list)):
        p2 = np.array(p2)
    if isinstance(p3, (tuple, list)):
        p3 = np.array(p3)

    points = list()
    for t in np.linspace(0, 1, inserted + 2):
        points.append(p0 * np.power((1 - t), 3) + 3 * p1 * t * np.power((1 - t), 2) + 3 * p2 * (1 - t) * np.power(t,
                                                                                                                  2) + p3 * np.power(
            t, 3))

    return np.vstack(points)


def smoothing_base_bezier(date_x, date_y, k=0.5, inserted=10, closed=False):
    """
    基于三阶贝塞尔曲线的数据平滑算法

    date_x  - x维度数据集，list或numpy.ndarray类型
    date_y  - y维度数据集，list或numpy.ndarray类型
    k   - 调整平滑曲线形状的因子，取值一般在0.2~0.6之间。默认值为0.5
    inserted - 两个原始数据点之间插值的数量。默认值为10
    closed  - 曲线是否封闭，如是，则首尾相连。默认曲线不封闭
    """

    assert isinstance(date_x, (list, np.ndarray)), u'x数据集不是期望的列表或numpy数组类型'
    assert isinstance(date_y, (list, np.ndarray)), u'y数据集不是期望的列表或numpy数组类型'

    if isinstance(date_x, list) and isinstance(date_y, list):
        assert len(date_x) == len(date_y), u'x数据集和y数据集长度不匹配'
        date_x = np.array(date_x)
        date_y = np.array(date_y)
    elif isinstance(date_x, np.ndarray) and isinstance(date_y, np.ndarray):
        assert date_x.shape == date_y.shape, u'x数据集和y数据集长度不匹配'
    else:
        raise Exception(u'x数据集或y数据集类型错误')

    # 第1步：生成原始数据折线中点集
    mid_points = list()
    for i in range(1, date_x.shape[0]):
        mid_points.append({
            'start': (date_x[i - 1], date_y[i - 1]),
            'end': (date_x[i], date_y[i]),
            'mid': ((date_x[i] + date_x[i - 1]) / 2.0, (date_y[i] + date_y[i - 1]) / 2.0)
        })

    if closed:
        mid_points.append({
            'start': (date_x[-1], date_y[-1]),
            'end': (date_x[0], date_y[0]),
            'mid': ((date_x[0] + date_x[-1]) / 2.0, (date_y[0] + date_y[-1]) / 2.0)
        })

    # 第2步：找出中点连线及其分割点
    split_points = list()
    for i in range(len(mid_points)):
        if i < (len(mid_points) - 1):
            j = i + 1
        elif closed:
            j = 0
        else:
            continue

        x00, y00 = mid_points[i]['start']
        x01, y01 = mid_points[i]['end']
        x10, y10 = mid_points[j]['start']
        x11, y11 = mid_points[j]['end']
        d0 = np.sqrt(np.power((x00 - x01), 2) + np.power((y00 - y01), 2))
        d1 = np.sqrt(np.power((x10 - x11), 2) + np.power((y10 - y11), 2))
        # print(d0, d1)
        k_split = 1.0 * d0 / (d0 + d1)

        mx0, my0 = mid_points[i]['mid']
        mx1, my1 = mid_points[j]['mid']

        split_points.append({
            'start': (mx0, my0),
            'end': (mx1, my1),
            'split': (mx0 + (mx1 - mx0) * k_split, my0 + (my1 - my0) * k_split)
        })

    # 第3步：平移中点连线，调整端点，生成控制点
    crt_points = list()
    for i in range(len(split_points)):
        vx, vy = mid_points[i]['end']  # 当前顶点的坐标
        dx = vx - split_points[i]['split'][0]  # 平移线段x偏移量
        dy = vy - split_points[i]['split'][1]  # 平移线段y偏移量

        sx, sy = split_points[i]['start'][0] + dx, split_points[i]['start'][1] + dy  # 平移后线段起点坐标
        ex, ey = split_points[i]['end'][0] + dx, split_points[i]['end'][1] + dy  # 平移后线段终点坐标

        cp0 = sx + (vx - sx) * k, sy + (vy - sy) * k  # 控制点坐标
        cp1 = ex + (vx - ex) * k, ey + (vy - ey) * k  # 控制点坐标

        if crt_points:
            crt_points[-1].insert(2, cp0)
        else:
            crt_points.append([mid_points[0]['start'], cp0, mid_points[0]['end']])

        if closed:
            if i < (len(mid_points) - 1):
                crt_points.append([mid_points[i + 1]['start'], cp1, mid_points[i + 1]['end']])
            else:
                crt_points[0].insert(1, cp1)
        else:
            if i < (len(mid_points) - 2):
                crt_points.append([mid_points[i + 1]['start'], cp1, mid_points[i + 1]['end']])
            else:
                crt_points.append([mid_points[i + 1]['start'], cp1, mid_points[i + 1]['end'], mid_points[i + 1]['end']])
                crt_points[0].insert(1, mid_points[0]['start'])

    # 第4步：应用贝塞尔曲线方程插值
    out = list()
    for item in crt_points:
        group = bezier_curve(item[0], item[1], item[2], item[3], inserted)
        out.append(group[:-1])

    out.append(group[-1:])
    out = np.vstack(out)

    return out.T[0], out.T[1]


#
# display the running path:
#
def show_orbit(orbits, rotate=False):
    # t.title("Orbits:")
    # t.setup(800, 600, 0, 0)
    x, y = np.array(orbits)[:, 0], np.array(orbits)[:, 1]

    plt.plot(x, y, 'ro')                    # ro: red circle
    x_curve, y_curve = smoothing_base_bezier(x, y, k=0.2, closed=False)
    # x_orbits, y_orbits = evaluate_bezier(np.array(orbits), 50)
    plt.plot(x_curve, y_curve)
    # plt.plot(x_curve, y_curve, label='$k=0.2$')
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')      # top-left corner is (0, 0)
    ax.invert_yaxis()
    ax.set(title="Orbits")
    plt.axis('off')
    # plt.legend(loc='best')
    plt.savefig('orbit_figure.png')         # not support .jpg when pyinstaller!

    plt.show()

    x_orbits = [x[0] for x in orbits]
    y_orbits = [x[1] for x in orbits]

    SCALE = 3
    EDGE = 50           # the edge width
    WIDTH = int(max(x_orbits) - min(x_orbits) + EDGE*2)
    HEIGHT = int(max(y_orbits) - min(y_orbits) + EDGE*2)

    # print(x_orbits, WIDTH)

    if rotate:
        WIDTH, HEIGHT = HEIGHT, WIDTH

    canvas = np.zeros((HEIGHT*SCALE, WIDTH*SCALE, 3), dtype="uint8")
    # print(canvas.shape, HEIGHT*SCALE, WIDTH*SCALE)
    last_x, last_y = 0, 0

    # Attention: video size(W, H) is not the JPEG's size(H, W)
    videoWriter = cv2.VideoWriter("Orbit.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 5, (WIDTH*SCALE, HEIGHT*SCALE))

    for i in range(len(orbits)):
        x, y = orbits[i]
        # print(i, orbits[i], x, y)

        if x != 0:
            x = int((x - min(x_orbits) + EDGE)*SCALE)
            y = int((y - min(y_orbits) + EDGE)*SCALE)

            # if too close, not display
            if last_x != 0:
                distance = sqrt((x-last_x)**2 + (y-last_y)**2)
                if distance <= 10:
                    # print(i, ": distance = ", distance, ", continue ...")
                    continue

            if rotate:
                cv2.circle(canvas, (y, x), 16, (0, 255, 255), -1)
                cv2.putText(canvas, str(i), (y-4*len(str(i)), x+4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                if last_x != 0:
                    cv2.line(canvas, (last_y, last_x), (y, x), (0, 255, 255), thickness=1, lineType=cv2.LINE_8)

            else:
                cv2.circle(canvas, (x, y), 16, (0, 255, 255), -1)
                cv2.putText(canvas, str(i), (x-4*len(str(i)), y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                # 是否联线：
                # if last_x != 0:
                #     cv2.line(canvas, (last_x, last_y), (x, y), (0, 255, 255), thickness=1, lineType=cv2.LINE_8)

            last_x, last_y = x, y

        cv2.imshow("Orbit", canvas)  # 16
        videoWriter.write(canvas)
        cv2.waitKey(200)  # 17

    print("轨迹动画保存在orbit.mp4, 轨迹图片为orbit_figure.png。")
    return

#
# Get the coordinate according to color:
#
def get_coord_by_color(image_file, way='AUTO'):
    frame = cv2.imread(image_file)
    ROWS = frame.shape[0]
    COLS = frame.shape[1]
    # print(image_file, ROWS, COLS)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)    # 色彩空间转换为hsv，便于分离

    # HSV: color_pickup() could get the specify color's HSV value
    lower_hsv = np.array([15, 100, 100])  # 提取颜色的低值: Yellow
    high_hsv = np.array([30, 255, 255])  # 提取颜色的高值

    # Create a mask:
    mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=high_hsv)

    # Find contours:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw all contours: some are noise!
    # cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    # Finding The Largest Contour:
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    # print(len(contour_sizes))

    if len(contour_sizes) > 0:
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
        # cv2.drawContours(frame, biggest_contour, -1, (0, 255, 0), 3)

        # Bounding Rectangle
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
        x, y, w, h = cv2.boundingRect(biggest_contour)

        # Get the coordinate of rectangle's center:
        cv2.circle(frame, (int(x + w / 2), int(y + h / 2)), 2, (0, 255, 0), 1)
    else:
        # not find yellow block:
        h = w = 0

    # print(x, y, h, w, ROWS, COLS)
    # not too small or not near border:
    if way != 'AUTO':       # 人工指定
        print("[%s]：绿点为自动判断的中心位置，若无需修改按ESC退出。若被遮挡或位置有偏差，请在指定位置双击鼠标左键后按ESC退出..." % image_file)

        # mouse callback:
        def draw_circle(event, x, y, flags, param):
            # global point
            if event == cv2.EVENT_LBUTTONDBLCLK:
                cv2.circle(frame, (x, y), 12, (0, 255, 255), -1)

                # How to return [x,y] ?
                param = [x, y]
                # return x, y
            elif event == cv2.EVENT_RBUTTONDOWN:            # no function！
                print("delete current jpeg: ", param[1])
                param[0] = [1, 2]

        cv2.namedWindow('image')
        point = [-1, -1]
        cv2.setMouseCallback('image', draw_circle, [point, image_file])

        while 1:
            cv2.imshow('image', frame)
            if cv2.waitKey(20) & 0xFF == 27:    # for 64bit OS
                break

        cv2.destroyAllWindows()
        cv2.imwrite(image_file, frame)

        return [0, 0]

    elif h < 10 or w < 10 or x < COLS*0.2 or x > COLS*0.8 or y < ROWS*0.2 or y > ROWS*0.8:      # not found
        return [0, 0]

    else:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)  # draw yellow rect: BGR
        cv2.imwrite(image_file, frame)

        return [int(x+w/2), int(y+h/2)]

#
# Get all coordinate at image path:
#
def get_all_coords(image_path, way='AUTO'):
    circleOrbit = []

    for file in os.listdir(image_path):
        image_file = os.path.join(image_path, file)

        coord = get_coord_by_color(image_file, way)
        # print(coord)

        if coord != [0, 0]:
            circleOrbit.append(coord)

    return circleOrbit

#
# Get video's FPS(Frame per second)
#
def get_fps(video_file):
    video = cv2.VideoCapture(video_file)

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        # print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
        # print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    video.release()
    return int(fps)

#
# generate image file from video:
#
def gen_image(video_file, image_path, start=0, framenum=60, interval=1):
    times = 0
    seq = 1

    if not os.path.isfile(video_file):
        print("请检查文件%s是否存在 ..." % video_file)
        return

    # 提取视频的频率
    frameFrequency = interval
    startFrame = start * get_fps(video_file)
    # print("FPS: ", get_fps(video_file))

    if not os.path.exists(image_path):
        # 如果文件目录不存在则创建目录
        os.makedirs(image_path)

    print("从视频文件[%s]第%d秒开始，间隔%d帧，共提取%d帧 ..." % (video_file, start, interval, framenum))
    camera = cv2.VideoCapture(video_file)

    while True:
        times += 1
        res, image = camera.read()
        if not res:
            # print('not res , not image')
            break
        # print(times)
        if times >= (startFrame+framenum*interval):
            break
        elif times >= startFrame:
            if (times-startFrame) % frameFrequency == 0:
                image_file = '%s%03d%s' % (image_path, seq, '.jpg')
                cv2.imwrite(image_file, image)
                print(image_file)
                seq += 1

    camera.release()
    print("图片生成成功，保存在%s ..." % image_path)

    return

#
# Get color's BGR/GRAY/HSV according to JPEG
#
def color_pickup(imgfile, color_format="HSV"):
    #
    def mouseColor(event, x, y, flags, param):
        # print(param[0], param[1])
        if event == cv2.EVENT_LBUTTONDOWN:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            print("BGR: ", img[y, x], "GRAY: ", gray[y, x], "HSV: ", hsv[y, x])     # (x,y)'s HSV
            param[1] = [x, y]

    # path, out = input('请输入图片名称或路径，再空格输入选择的颜色格式（bgr/gray/hsv）\n').split()
    img = cv2.imread(imgfile)      # read in BGR format

    cv2.namedWindow("Color Picker")
    a = []
    cv2.setMouseCallback("Color Picker", mouseColor, [img, a])
    cv2.imshow("Color Picker", img)

    if cv2.waitKey(0):
        cv2.destroyAllWindows()

    print(a)


if __name__ == '__main__':
    video_file = './circle_hd.mp4'
    image_path = './Picture/'

    # color_pickup('../DATA/Picture/008.jpg', color_format='HSV')
    get_fps(video_file)

    while 1:
        print("\n           运动轨迹提取\n\n"
              "1. 从视频文件逐帧提取数据，生成图片文件；\n"
              "2. 手工指定每张图片黄块的位置（位置更精准，该步骤可选）；\n"
              "3. 自动定位图片文件中的内圆黄点，显示黄点运动轨迹；\n"
              "4. 显示帮助；\n"
              "5. 退出！\n\n")
        choice = input("请输入选项[1-5]: ")

        if choice == '1':
            # 1.Gen image:
            if os.path.isdir(image_path):
                for img_file in os.listdir(image_path):
                    # print(os.path.join(image_path, img_file))
                    os.remove(os.path.join(image_path, img_file))

            choice = input("请指定开始秒数，总帧数和间隔帧数，用空格分隔，例如'2 100 1'。直接回车从第2秒开始，每3帧共提取60张图片：")
            if choice == "":
                gen_image(video_file, image_path, start=2, framenum=60, interval=3)
            else:
                option = choice.split(" ")
                if len(option) != 3:
                    print("请检查输入格式 ...")
                else:
                    gen_image(video_file, image_path, start=int(option[0]), framenum=int(option[1]), interval=int(option[2]))
        elif choice == '2':
            # 2.You can edit the yellow color:
            circleOrbit = get_all_coords(image_path, way='MANUAL')
        elif choice == '3':
            # 2.You can edit the yellow color:
            print("自动提取黄点的坐标位置，忽略被遮挡的位置 ...")
            circleOrbit = get_all_coords(image_path)
            # print("The orbit is: ", circleOrbit)
            # show_orbit(circleOrbit, rotate=True)
            show_orbit(circleOrbit)
            cv2.waitKey(3000)               # 3s
            cv2.destroyAllWindows()         # close opencv window
        elif choice == '4':
            print("\n将视频文件更名为circle_hd.mp4, 放在当前目录，确保当前目录下有opencv_videoio_ffmpeg412_64.dll文件。\n"
                  "1. 从视频文件生成图片，按提示输入，图片在.\Picture目录下；\n"
                  "2. 由于可能有遮挡，可手工指定黄块位置，若系统自动判断的位置正确(小绿点)，可直接ESC退出；\n"
                  "3. 也可直接自动判断黄块位置，忽略被遮挡的情况，生成轨迹图片和运动视频，保存为orbit_figure.png和Orbit.mp4")

        elif choice == '5' or choice == 'q':
            break
        else:
            print("未知选项，请重新输入!")