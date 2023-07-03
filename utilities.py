import torch
from torchvision import transforms
import time
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
import argparse
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import re
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import pickle


def csv_converter(path, fname):
    df = pd.read_csv(path + fname, sep=";")
    df_ = pd.DataFrame(data=df)
    all_coordinates = []
    all_poses = []
    headers_names = df_.columns.values
    for col_num in range(len(df_)):
        # print(df_[headers_names[col_num]].values)
        # print(df_.loc[col_num].values)
        all_poses.append([df_.loc[col_num].values[2]])
        all_coordinates.append([])
        for i in range(3, len(df_.loc[col_num].values)):
            nums = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", df_.loc[col_num].values[i])
            for num in nums:
                all_coordinates[col_num].append(float(num))

    return all_poses, all_coordinates


def pose_to_num(poses_):
    poses_list = ["walk", "fall", "fallen", "sitting"]
    all_poses_num = []
    for pose in poses_:
        if pose[0] == "walk":
            all_poses_num.append(["0"])
        if pose[0] == "fall":
            all_poses_num.append(["1"])
        if pose[0] == "fallen":
            all_poses_num.append(["2"])
        if pose[0] == "sitting":
            all_poses_num.append(["3"])

    return all_poses_num


def get_pose_from_num(pose_number):
    if pose_number[0] == "0":
        return "walk"
    if pose_number[0] == "1":
        return "fall"
    if pose_number[0] == "2":
        return "fallen"
    if pose_number[0] == "3":
        return "sitting"
    else:
        return "code_error"


def most_frequent(List):
    counter = 0
    num = List[0]
    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency >= counter):
            counter = curr_frequency
            num = i
    if counter == 1:
        return List[-1]
    return num


def keypoints_parser(kps, dt_line):
    human = kps[0]
    for points in human:
        dt_line.append((round(points[0], 2), round(points[1], 2)))
    return dt_line


def get_coords_line(kps):
    coords_line = []
    for kp in kps:
        coords_line.append(kp[0])
        coords_line.append(kp[1])
    return coords_line

