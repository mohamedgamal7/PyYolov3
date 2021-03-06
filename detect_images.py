from __future__ import division
import time
from torch.autograd import Variable
import cv2
from utils import *
import argparse
import os
import os.path as osp
from model import Darknet
import pickle as pkl
import pandas as pd
import random


def arg_parse():
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--images", dest='images', help="An image or directory to perform predictions upon",
                        default="images", type=str)
    parser.add_argument("--det", dest='detections', help="An image or firectory to store detections to",
                        default="deteections", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Objectness score", default=0.9)
    parser.add_argument("--nms", dest="nms_thresh", help="NMS Threshhold", default=0.1)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file",
                        default="yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile"
                        , default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)

    return parser.parse_args()


# --------------------------------------------------------------------
# load coco names
def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names


# --------------------------------------------------------------------
# resize image with unchanged aspect ratio using padding
def letterbox_image(img, inp_dim):
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))  # resize keeping ap ratio
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image
    return canvas


# -----------------------------------------------------------
# prepare the image for the model
def prep_image(img, inp_dim):
    img = cv2.resize(img, (inp_dim, inp_dim))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img


# -----------------------------------------------------------
# return bounding box and text on an image
def write_bbox(x, results):
    c11, c12 = x[1:3]
    c1 = int(c11), int(c12)
    c21, c22 = x[3:5]
    c2 = int(c21), int(c22)
    img = results[int(x[0])]
    cls = int(x[-1])
    classes = load_classes("data/coco.names")
    label = "{0}".format(classes[cls])
    colors = pkl.load(open("pallete", "rb"))
    color = random.choice(colors)  # choose a random color for each class in bounding box
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);
    return img


# -----------------------------------------------------------
# detects bounding boxes and classes on images
def main_detector():
    classes = load_classes("data/coco.names")
    args = arg_parse()
    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = True
    num_classes = 80  # COCO number of classes

    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    model.eval()

    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print("No file or directory with the name {}".format(images))
        exit()

    if not os.path.exists(args.detections):
        os.makedirs(args.detections)

    load_batch = time.time()
    loaded_ims = [cv2.imread(x) for x in imlist]

    # PyTorch Variables for images
    im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))

    # List containing dimensions of original images
    im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

    if CUDA:
        im_dim_list = im_dim_list.cuda()

    leftover = 0
    if (len(im_dim_list) % batch_size):
        leftover = 1

    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover
        im_batches = [torch.cat((im_batches[i * batch_size: min((i + 1) * batch_size,
                                                                len(im_batches))])) for i in range(num_batches)]

    write = 0

    for i, batch in enumerate(im_batches):
        # load the image

        if CUDA:
            batch = batch.cuda()
        with torch.no_grad():
            prediction = model(Variable(batch), CUDA)

        prediction = write_results(prediction, confidence, num_classes, nms_conf=nms_thesh)

        if type(prediction) == int:

            for im_num, image in enumerate(imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]):
                im_id = i * batch_size + im_num
            continue

        prediction[:, 0] += i * batch_size  # transform the atribute from index in batch to index in imlist

        if not write:  # If we haven't initialised output
            output = prediction
            write = 1
        else:
            output = torch.cat((output, prediction))

        for im_num, image in enumerate(imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]):
            im_id = i * batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        if CUDA:
            torch.cuda.synchronize()

    try:
        output
    except NameError:
        print("No detections were made")
        exit()

    im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

    scaling_factor = torch.min(inp_dim / im_dim_list, 1)[0].view(-1, 1)
    output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
    output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2
    output[:, 1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
        output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])

    list(map(lambda x: write_bbox(x, loaded_ims), output))
    det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.detections, x.split("/")[-1]))
    list(map(cv2.imwrite, det_names, loaded_ims))
    end = time.time()
    print("Inference time per image is :{:5.2f}".format((end - load_batch) / len(imlist)))
    torch.cuda.empty_cache() # to prenvent cuda out of memory


if __name__ == "__main__":
    main_detector()
