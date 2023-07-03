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
import csv
from data_lists import train_data, test_data
from utilities import keypoints_parser

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Lines 19-66 are from YOLOv7 model a bit modified
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
    print("#################")

    for idx in range(output.shape[0]):
        human_data_line = []
        for iter in range(17):
            human_data_line.append((output[idx, 7:].T[0 + 3 * iter], output[idx, 7:].T[1 + 3 * iter]))
        ret_kps.append(human_data_line)
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

    return nimg, ret_kps

model = load_model()

def pose_estimation_video(data_path, markup):
    # cap = cv2.VideoCapture(filename)
    # VideoWriter for saving the video
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter("fname.mp4", fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    # Vid_counter can be changed differing on type of data (test up to 18 or train up to 30)
    # you are using from data_lists
    vid_counter = 18
    csvname = '7test_video.csv' # str(vid_counter) + 'vid_data_test.csv'
    with open(csvname, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';')
        spamwriter.writerow(['time', "vname", 'pose', 'bp0', 'bp1', 'bp2', 'bp3', 'bp4', 'bp5', 'bp6', 'bp7', 'bp8',
                             'bp9', 'bp10', 'bp11', 'bp12', 'bp13', 'bp14', 'bp15', 'bp16'])

    # Cycle for each chosen video
    for video_n in range(vid_counter):
        vid_path = data_path + markup[video_n][3]
        print(vid_path)
        strange_falls = ["50wtf9.mp4", "50wtf12.mp4", "50wtf16.mp4",
                         "50wtf28.mp4", "50wtf31.mp4", "50wtf47.mp4", "50wtf49.mp4"]
        if markup[video_n][3] not in strange_falls:
            continue
        cap = cv2.VideoCapture(vid_path)
        frame_count = 0  # to count total frames
        total_fps = 0  # to get the final frames per second
        fps_time = 0
        frame_n = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        pose_label = "none"

        # Cycle for each frame of video
        while cap.isOpened():
            (ret, frame) = cap.read()
            if ret:
                frame_number = frame_n / vid_fps
                frame_n += 1
                data_line = []
                data_line.append(round(frame_number, 2))
                data_line.append(vid_path)

                time_1 = markup[video_n][1]
                time_2 = markup[video_n][2]
                init_pose = markup[video_n][0] # "sitting" # "walk"
                pose_label = "none"
                if frame_number < time_1:
                    pose_label = init_pose
                if time_1 <= frame_number < time_2:
                    pose_label = "fall"
                if frame_number >= time_2:
                    pose_label = "fallen"
                data_line.append(pose_label)

                print(data_line)
                print(frame_n)
                print(vid_fps)
                print("#####")

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                output, frame = run_inference(frame)
                frame, keypoints_ = draw_keypoints(output, frame)

                # Getting human coordinates
                try:
                    data_line = keypoints_parser(keypoints_, data_line)
                except:
                    pass
                # out.write(frame)
                cv2.putText(frame,
                            "FPS: %f" % (1.0 / (time.time() - fps_time)),
                            (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)
                frame = cv2.resize(frame, (int(cap.get(3)), int(cap.get(4))))

                cv2.imshow('Pose estimation', frame)
                fps_time = time.time()

                if 23 > len(data_line) > 3:
                    # data_line[2] = "none"
                    with open(csvname, 'a', newline='') as csvfile:
                        spamwriter = csv.writer(csvfile, delimiter=';')
                        spamwriter.writerow(data_line)
            else:
                break

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        # out.release()
        cv2.destroyAllWindows()


# parser = argparse.ArgumentParser()
# parser.add_argument('-i', '--input', required=True,
#                     help='path to the input data')
# args = vars(parser.parse_args())
#
# pose_estimation_video(args['input'])  # "./videos/50wtf.mp4")  # args['input'])

data_path_ = "videos/cuts_test/"
pose_estimation_video(data_path_, test_data)
