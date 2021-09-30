import torch
import numpy as np


# ----------------------------------------------------------------------------
# Parse lines in a cfg files and outputs a list of lines  i.e (no empty lines ,comments or white spaces)
def create_lines_list(cfgfile):
    first_char_in_line = 0
    with open(cfgfile, 'r') as file:
        lines = [line.strip() for line in file if len(line.strip()) > 0 and line[first_char_in_line] != "#"]
    return lines


# ----------------------------------------------------------------------------
# create blocks representing the model layers and outputs a list of blocks
def create_blocks_list(lines):
    first_char_in_line = 0
    block = {}  # dict for each layer
    blocks = []  # list of blocks (layers)

    for line in lines:
        if line[first_char_in_line] == "[":  # indicates start of a new layer
            if len(block) != 0:
                blocks.append(block)  # store the previous block in blocks list
                block = {}  # empty the dicitinary for the next block
            block["type"] = line[1:-1]  # take the type of the block without []
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()  # remove redundant spaces
    blocks.append(block)  # append the last block

    return blocks


# ----------------------------------------------------------------------------
# Takes a cfg file and outputs a list of dicts representing each layer in the network
def parse_cfg(cfgfile):
    lines = create_lines_list(cfgfile)
    blocks = create_blocks_list(lines)
    return blocks


# ----------------------------------------------------------------------------
# takes an output of a yolo layer turns it into a 2-D tensor for each cell's bounding box , its attribuites
# useful for feature map concatenation at different scales
def transform_predictions(prediction, input_dim, anchors, num_classes, GPU=True):
    batch_size = prediction.size(0)
    stride = input_dim // prediction.size(2)  # if you divide input by output dimension  you get strides
    grid_size = input_dim // stride  # same as prediction.size(2)
    bbox_attrs = 5 + num_classes  # x,y,w,h,p score and num of classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)

    prediction = prediction.transpose(1, 2).contiguous()  # if reshape used this is not needed
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)
    anchors = [(anchor[0] / stride, anchor[1] / stride) for anchor in anchors]
    # scaled version of the input anchors by the stride value

    # The bounding box attributes are as follows :
    # tx ,ty ,tw,th and objectioness score
    # we will apply sigmoid to x,y outputs along with the p score
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)  # make a map of the grid (x,y) each point represents cell top location

    x_offset = torch.FloatTensor(a).view(-1, 1)  # x offests in a single vector
    y_offset = torch.FloatTensor(b).view(-1, 1)  # y offsets in a single vector

    if GPU:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    # repeat the offset for eaach anchor(3 anchors) and concat

    prediction[:, :, :2] += x_y_offset  # add offsets to the predicted  x,y cordinates

    # perform the log space transform height and the width as mentioned in the paper
    anchors = torch.FloatTensor(anchors)

    if GPU:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    # apply sigmoid to classes (sigmoid not softmax as softmax is mutually exlusive and this is not the case in obj
    # detection)
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))
    prediction[:, :, :4] *= stride  # resize detected features to size of input image

    return prediction


# gets unique elements of a tensor in a certain dimension
def unique(tensor):
    tensor_np = tensor.detach().cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


# ----------------------------------------------------------------------------
# iou of two bounding boxes
def bbox_iou(box1, box2):
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the co0rdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # intersection area
    # we add 1 as the indicies of cells start from 0
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1,
                                                                                     min=0)

    # union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


# ----------------------------------------------------------------------------
# returns predictions as true bounding box
def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    # predictions above objectness score  only are considered
    prediction = prediction * conf_mask  # zero out predictions below confidence score
    box_corner = prediction.new(prediction.shape)  # same data type and shape as prediction
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)  # top corner x
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)  # top corner y
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)  # bottom corner x
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)  # bottom corner y
    prediction[:, :, :4] = box_corner[:, :, :4]

    batch_size = prediction.size(0)
    is_intialized = False
    # we need to loop on each image alone as number of predictions (bboxes) is different from image to another
    for ind in range(batch_size):
        image_pred = prediction[ind]  # one image only
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)  # max class index and score
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)  # concat the tuple of bboxes , max class index and score

        non_zero_ind = (torch.nonzero(image_pred[:, 4]))  # remove all predictions below the objectioness score
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
        except:
            continue

        if image_pred_.shape[0] == 0:  # no prediction skip to the next image
            continue
        img_classes = unique(image_pred_[:, -1])  # unique image classes in the same image
        for cls in img_classes:
            # get the detections with one particular class
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[
                1]  # sort classes according to prediction score
            image_pred_class = image_pred_class[conf_sort_index]  # sorted class
            idx = image_pred_class.size(0)  # Number of detections

            for i in range(idx):
                # fet the IOUs of all boxes that come after the one we are looking at
                # in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])
                except ValueError:
                    break

                except IndexError:
                    break

                # zero out all the detections that have IoU < treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i + 1:] *= iou_mask

                # remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)
                batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)  # index of image in batch
                # repeat the batch_id for as many detections of the class cls in the image
                seq = batch_ind, image_pred_class

                if not is_intialized:
                    output = torch.cat(seq, 1)
                    is_intialized = True
                else:
                    out = torch.cat(seq, 1)
                    output = torch.cat((output, out))

    try:
        return output
    except:
        return 0


# ----------------------------------------------------------------------------
# load coco names
def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names
