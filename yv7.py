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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def load_model():
    model = torch.load('yolov7-w6-pose.pt', map_location=device)['model']
    # Put in inference mode
    model.float().eval()

    if torch.cuda.is_available():
        # half() turns predictions into float16 tensors
        # which significantly lowers inference time
        model.half().to(device)
    return model


model = load_model()


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
    # print(output.shape[0])
    for idx in range(output.shape[0]):
        # print(output[idx, 7:].T)
        human_data_line = []
        for iter in range(17):
            human_data_line.append(output[idx, 7:].T[0 + 3 * iter])
            human_data_line.append(output[idx, 7:].T[1 + 3 * iter])
        ret_kps.append(human_data_line)

        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
    print(ret_kps)
    return nimg


def pose_estimation_video(filename):
    cap = cv2.VideoCapture(filename)
    # VideoWriter for saving the video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter("50wtf_yv7.mp4", fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    frame_count = 0  # to count total frames
    total_fps = 0  # to get the final frames per second
    fps_time = 0
    frame_n = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    pose_label = "none"

    while cap.isOpened():
        (ret, frame) = cap.read()
        if ret:
            frame_number = frame_n / fps
            frame_n += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output, frame = run_inference(frame)
            frame = draw_keypoints(output, frame)

            cv2.putText(frame,
                        "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            frame = cv2.resize(frame, (int(cap.get(3)), int(cap.get(4))))
            out.write(frame)
            cv2.imshow('Pose estimation', frame)
            fps_time = time.time()
        else:
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True,
                    help='path to the input data')
args = vars(parser.parse_args())

pose_estimation_video(args['input'])  # "./videos/50wtf.mp4")  # args['input'])
