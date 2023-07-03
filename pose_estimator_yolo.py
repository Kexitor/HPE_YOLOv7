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
from utilities import csv_converter, pose_to_num, get_pose_from_num, most_frequent, keypoints_parser, get_coords_line

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.is_available())

# Lines 25-73 are from YOLOv7 model a bit modified
def load_model():
    model = torch.load('yolov7-w6-pose.pt', map_location=device)['model']
    # Put in inference mode
    model.float().eval()

    if torch.cuda.is_available():
        # half() turns predictions into float16 tensors
        # which significantly lowers inference time
        model.half().to(device)
    return model


def run_inference(image):
    # Resize and pad image
    image = letterbox(image, 960, stride=64, auto=True)[0]  # shape: (567, 960, 3)
    # Apply transforms
    image = transforms.ToTensor()(image)  # torch.Size([3, 567, 960])
    if torch.cuda.is_available():
        image = image.half().to(device)
    # Turn image into batch
    image = image.unsqueeze(0)  # torch.Size([1, 3, 567, 960])
    with torch.no_grad():
        output, _ = model(image)
    return output, image


def draw_keypoints(output, image):
    ret_kps = []
    output = non_max_suppression_kpt(output,
                                     0.25,  # Confidence Threshold
                                     0.65,  # IoU Threshold
                                     nc=model.yaml['nc'],  # Number of Classes
                                     nkpt=model.yaml['nkpt'],  # Number of Keypoints
                                     kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

    for idx in range(output.shape[0]):
        human_data_line = []
        for iter in range(17):
            human_data_line.append((output[idx, 7:].T[0 + 3 * iter], output[idx, 7:].T[1 + 3 * iter]))
        ret_kps.append(human_data_line)
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

    return nimg, ret_kps


model = load_model()


def pose_estimation_video(filename, NN_pose_est):
    vid_path = filename
    # print(vid_path)
    cap = cv2.VideoCapture(vid_path)
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter("50wtf_yv7.mp4", fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    frame_count = 0  # to count total frames
    total_fps = 0  # to get the final frames per second
    fps_time = 0
    frame_n = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_fps = cap.get(cv2.CAP_PROP_FPS)

    # Main cycle for each frame
    while cap.isOpened():
        (ret, frame) = cap.read()
        if ret:
            frame_number = frame_n / vid_fps
            frame_n += 1
            data_line = []
            data_line.append(round(frame_number, 2))
            data_line.append(vid_path)

            pose_label = "none"

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output, frame = run_inference(frame)
            frame, keypoints_ = draw_keypoints(output, frame)

            # Classifying pose for identified human, can classify several humans poses in frame
            coords_line = []
            try:
                coords_line = [get_coords_line(keypoints_[0])]
                for human_kps in keypoints_:
                    # print(human_kps)
                    hum_crd_ln = [get_coords_line(human_kps)]
                    if 34 >= len(hum_crd_ln) >= 1:
                        pose_code = NN_pose_est.predict(hum_crd_ln)
                        pose_label = get_pose_from_num(pose_code)
                        cv2.putText(frame,
                                    "pose: %s" % (pose_label),
                                    (int(hum_crd_ln[0][0]), int(hum_crd_ln[0][1]) - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 0, 255), 2)

            except:
                pass

            pose_label = "none"

            if 34 >= len(coords_line) >= 1:
                pose_code = NN_pose_est.predict(coords_line)
                pose_label = get_pose_from_num(pose_code)

            cv2.putText(frame,
                        "NN: %s" % (pose_label),
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

            cv2.putText(frame,
                        "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            frame = cv2.resize(frame, (int(cap.get(3)), int(cap.get(4))))
            # out.write(frame)
            cv2.imshow('Pose estimation', frame)
            fps_time = time.time()
        else:
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    # out.release()
    cv2.destroyAllWindows()


# Getting train dataset
path = ""  # "videos/csv_files/"
filename = "37vid_data_train_yolo.csv"# "37vid_data_train.csv" "37vid_data_train.csv"
train_poses, train_coords = csv_converter(path, filename)
train_poses_num = pose_to_num(train_poses)

# Training model

# NN = MLPClassifier(solver='lbfgs', activation='logistic', alpha=1e-5, hidden_layer_sizes=(150, 10), random_state=1,
#                    max_iter=10000).fit(train_coords, train_poses_num)

# pkl_filename = "pickle_model37VTrD.pkl"
# with open(pkl_filename, 'wb') as file:
#     pickle.dump(NN, file)
NN = ""
with open("pickle_model37VTrD.pkl", 'rb') as file:
    NN = pickle.load(file)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True,
                    help='path to the input data')
args = vars(parser.parse_args())

pose_estimation_video(args['input'], NN)  # "./videos/50wtf.mp4")  # args['input'])

# data_path_ = "videos/cuts_test/"
# pose_estimation_video(data_path_, test_data)
