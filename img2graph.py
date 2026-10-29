import argparse
import math
import os
import cv2
import numpy as np
from operator import add
from functools import reduce
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import time
from collections import Counter
import networkx as nx
from collections import Counter
import scipy.sparse as sp
import itertools
import random

random.seed(0)


def get_grad(img, dir=None):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobely = cv2.convertScaleAbs(sobely)
    sobelxy2 = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

    return sobelxy2


def cal_bound(img, center, Rx, Ry):
    h, w = img.shape
    left = center[1] - Rx
    right = center[1] + Rx
    up = center[0] - Ry
    down = center[0] + Ry

    if left < 0:
        left = 0
    if right >= w:
        right = w - 1
    if up < 0:
        up = 0
    if down >= h:
        down = h - 1

    return int(left), int(right), int(up), int(down)


def center_select(img_grad, img_label):
    minPix_xy = np.where(img_grad == img_grad.min())

    while True:
        pos = random.randint(0, len(minPix_xy[0]) - 1)
        # print(len(minPix_xy[0]))
        if (img_label[minPix_xy[0][pos], minPix_xy[1][pos]] != 1):
            return [minPix_xy[0][pos], minPix_xy[1][pos]]


def cal_Radius_1(img, center, purity, threshold, var_threshold):
    Rx = 0
    Ry = 0

    flag = True
    flag_x = True
    flag_y = True
    item_count = 0
    temp_pixNum = 0

    center_value = int(img[center[0], center[1]])

    while True:

        if flag_x == True and flag_y == True:
            item_count += 1
        else:
            if flag_x:
                item_count = 1
            if flag_y:
                item_count = 2

        # print(item_count)

        if flag_x and item_count % 2 != 0:
            Rx += 1

        if flag_y and item_count % 2 == 0:
            Ry += 1

        left, right, up, down = cal_bound(img, center, Rx, Ry)
        pixNum = (down - up + 1) * (right - left + 1)

        if pixNum == temp_pixNum:
            return Rx, Ry, 1


        count = len(np.where(abs(np.int_(img[up:down + 1, left:right + 1]) - center_value) > threshold)[0])


        temp_purity = 1 - count / pixNum
        var = np.var(img[up:down + 1, left:right + 1])
        temp_pixNum = pixNum

        if temp_purity > purity and var < var_threshold:
            if purity < 0.99:
                purity = purity * 1.005
            else:
                purity = 0.99
            flag = True
        else:
            flag = False

        if flag == False and item_count % 2 != 0:
            flag_x = False
            Rx -= 1

        if flag == False and item_count % 2 == 0:
            flag_y = False
            Ry -= 1

        # print(flag_x, flag_y)

        if flag_x == False and flag_y == False:
            return Rx, Ry, temp_purity


def calulate_weight(img, center_1, center_2):
    left_1, right_1, up_1, down_1 = cal_bound(img, center_1[0], center_1[1], center_1[2])
    left_2, right_2, up_2, down_2 = cal_bound(img, center_2[0], center_2[1], center_2[2])

    x_list = [up_1, up_2, down_1, down_2]
    y_list = [left_1, left_2, right_1, right_2]
    x_list.sort()
    y_list.sort()

    res = (x_list[2] - x_list[1] + 1) * (y_list[2] - y_list[1] + 1)

    return res


def calulate_A_and_B(img, center_1, center_2):
    x1, x2, y1, y2 = cal_bound(img, center_1[0], center_1[1], center_1[2])
    x3, x4, y3, y4 = cal_bound(img, center_2[0], center_2[1], center_2[2])

    return (x2 - x1 + 1) * (y2 - y1 + 1) + (x4 - x3 + 1) * (y4 - y3 + 1)



def granular_balls_generate(img, purity_1=0.9, threshold=10, var_threshold_1=20):
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img_label = np.zeros(img.shape)
    img_grad = get_grad(img)
    max_Grad = img_grad.max()
    # print(grad_median)
    h, w = img.shape
    center = []
    center_count = 0

    start = time.time()
    while 0 in img_label:
        temp_center = center_select(img_grad, img_label)

        Rx, Ry, temp_purity = cal_Radius_1(img, temp_center, purity_1, threshold, var_threshold_1)

        left, right, up, down = cal_bound(img, temp_center, Rx, Ry)

        img_mean = img.mean()


        cx, cy = temp_center
        patch = img[max(cx - 1, 0): min(cx + 2, h), max(cy - 1, 0): min(cy + 2, w)]
        patch_flat = patch.flatten().astype(np.float32)
        patch_padded = np.pad(patch_flat, (0, 9 - len(patch_flat)), mode='constant', constant_values=0)

        ball_pixel = img[cx, cy]
        max_ball_pixel = img[up:down + 1, left:right + 1].max()
        min_ball_pixel = img[up:down + 1, left:right + 1].min()
        ave_ball_pixel = img[up:down + 1, left:right + 1].mean()

        center.append((
            temp_center,
            Rx, Ry,
            img_mean,
            img_grad[temp_center[0], temp_center[1]],
            max_ball_pixel, min_ball_pixel, ave_ball_pixel,

            ball_pixel,
            *patch_padded,
            math.sqrt(cx ** 2 + cy ** 2),
            math.atan2(cy, cx),
            ball_pixel / (ave_ball_pixel + 1e-5),
            img_mean - ave_ball_pixel,
            cx * ball_pixel,
            cy * img_mean,
        ))

        img_label[up:down + 1, left:right + 1] = 1

        img_grad[up:down + 1, left:right + 1] = max_Grad

        center_count += 1

    end = time.time()

    g = nx.Graph()

    for i in range(len(center)):
        g.add_node(str(i))


    for i in range(len(center)):
        for j in range(i + 1, len(center)):
            center_1 = center[i]
            center_2 = center[j]

            if (abs(center_1[0][0] - center_2[0][0]) - 1) < center_1[2] + center_2[2] and (
                    abs(center_1[0][1] - center_2[0][1]) - 1) < center_1[1] + center_2[1]:
                g.add_edge(str(i), str(j))

    a = nx.to_numpy_array(g)

    adj = a
    adj = sp.coo_matrix(adj)
    adj = np.vstack((adj.row, adj.col))

    center_array = np.zeros((len(center), 25))
    edge_attr = np.zeros((len(adj[0]), 3))
    center_ = np.zeros((len(center), 4))

    for i in range(len(adj[0])):

        center_1 = center[adj[0][i]]
        center_2 = center[adj[1][i]]

        temp = calulate_weight(img, center_1, center_2)
        temp_iou = calulate_A_and_B(img, center_1, center_2)

        center_dis = math.sqrt((center_1[0][0] - center_2[0][0]) ** 2 + (center_1[0][1] - center_2[0][1]) ** 2)
        edge_attr[i, :3] = [temp, temp / temp_iou, center_dis]

    edge_attr[:, 0] = edge_attr[:, 0] / edge_attr[:, 0].max()
    edge_attr[:, 2] = edge_attr[:, 2] / edge_attr[:, 2].max()



    radius = np.array([c[-6] for c in center])
    angle = np.array([c[-5] for c in center])
    pixel_ratio = np.array([c[-4] for c in center])
    global_diff = np.array([c[-3] for c in center])
    x_pixel_interact = np.array([c[-2] for c in center])
    y_global_interact = np.array([c[-1] for c in center])

    radius = normalize(radius)
    angle = (angle + np.pi) / (2 * np.pi)
    pixel_ratio = normalize(pixel_ratio)
    global_diff = normalize(global_diff)
    x_pixel_interact = normalize(x_pixel_interact)
    y_global_interact = normalize(y_global_interact)



    for id in range(len(center)):

        center_array[id, :19] = [center[id][0][0], center[id][0][1], center[id][1], center[id][2], center[id][3],center[id][4],
                            center[id][5], center[id][6], center[id][7], center[id][8],
                            center[id][9], center[id][10],center[id][11], center[id][12],center[id][13], center[id][14],center[id][15], center[id][16],center[id][17]]

        center_array[id, 19:] = [
            radius[id],
            angle[id],
            pixel_ratio[id],
            global_diff[id],
            x_pixel_interact[id],
            y_global_interact[id],
        ]


        center_[id] = [center[id][0][0], center[id][0][1], center[id][1], center[id][2]]


    return center_array, adj, edge_attr, center_


def add_gray_diff_feature(img, center_1, center_2, edge_attr, idx):
    x1, x2, y1, y2 = cal_bound(img, center_1[0], center_1[1], center_1[2])
    x3, x4, y3, y4 = cal_bound(img, center_2[0], center_2[1], center_2[2])
    mean_gray_1 = img[x1:x2+1, y1:y2+1].mean()
    mean_gray_2 = img[x3:x4+1, y3:y4+1].mean()
    gray_diff = abs(mean_gray_1 - mean_gray_2)
    edge_attr[idx, 3] = gray_diff / 255.0
    return edge_attr



def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)


def visualize_pixel_matrix(pixel_matrix, title='gray Image'):
    plt.figure(figsize=(6, 3))
    plt.imshow(pixel_matrix, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()
def show_image(image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('on')
    plt.show()
