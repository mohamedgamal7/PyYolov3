from __future__ import division
import time
from utils import *
from model import Darknet
import random
import pickle as pkl
import argparse
import cv2
from torch.autograd import Variable


# ------------------------------------------------------------------------
# parse arguments from the command line
def arg_parse():
    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')

    parser.add_argument("--video", dest='video', help="Video to run detection upon", default="vid.mp4", type=str)

    parser.add_argument("--dataset", dest="dataset", help="Dataset on which the network has been trained",
                        default="pascal")
    parser.add_argument("--confidence", dest="confidence",
                        help="Object Confidence to filter predictions", default=0.6)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.1)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file", default="yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile", default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)
    return parser.parse_args()


# -------------------------------------------------------------------------------
# keep original aspect ratio of the image
def letterbox_image_video(img, inp_dim):
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas


# ----------------------------------------------------------------------------
# get the image ready for the model
def prep_image_video(img, inp_dim):
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image_video(orig_im, (inp_dim, inp_dim)))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


# --------------------------------------------------------------------
# draws the bounding box and writes text on img
def write_on_img(x, img, colors):
    c11, c12 = x[1:3]
    c1 = int(c11), int(c12)
    c21, c22 = x[3:5]
    c2 = int(c21), int(c22)
    cls = int(x[-1])
    classes = load_classes('data/coco.names')
    label = "{0}".format(classes[cls])
    if label == "car":
        color = (0, 0, 255)
    elif label == "bus":
        color = (0, 255, 0)
    elif label == "person":
        color = (255, 0, 0)
    elif label == "traffic light":
        color = (255, 0,232)
    elif label == "truck":
        color = (255, 0,155)
    else:
        color = random.choice(colors)
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);
    return img


# ------------------------------------------------------------------------
# main video detector
def video_detector():

    args = arg_parse()
    videofile = args.video
    num_classes = 80

    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)

    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    CUDA = torch.cuda.is_available()
    if CUDA:
        model.cuda()
    model.eval()
    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    cap = cv2.VideoCapture(videofile)
    assert cap.isOpened(), 'Cannot capture source'
    frames = 0
    start = time.time()
    while cap.isOpened():

        ret, frame = cap.read()
        if ret:
            img, orig_im, dim = prep_image_video(frame, inp_dim) # prepare each frame for the model

            im_dim = torch.FloatTensor(dim).repeat(1, 2)

            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()

            with torch.no_grad():
                output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms_conf=nms_thesh)

            if type(output) == int:
                frames += 1
                print("FPS : {:5.2f}".format(frames / (time.time() - start)))
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('1'): #press 1 to esc
                    break
                continue

            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(inp_dim / im_dim, 1)[0].view(-1, 1)

            # transform bounding boxes to original image dimensions
            output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
            output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

            output[:, 1:5] /= scaling_factor # rescaling bounding boxes to original image dimensions

            # clip bounding boxes outside image
            for i in range(output.shape[0]):
                output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
                output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

            colors = pkl.load(open("pallete", "rb"))
            list(map(lambda x: write_on_img(x, orig_im, colors), output))

            cv2.imshow("frame", orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('1'):
                break
            frames += 1
            print("FPS : {:5.2f}".format(frames / (time.time() - start)))

        else:
            break


if __name__ == '__main__':
    video_detector()
