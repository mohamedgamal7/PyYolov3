import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from utils import transform_predictions ,parse_cfg
import cv2


# ----------------------------------------------------------------------------
# create pytorch modules out of the described blocks
def create_modules(blocks_):
    net_block = 0  # index of hyper parameters block
    net_info = blocks_[net_block]
    module_list = nn.ModuleList()  # to contain all parameters of nn.module
    prev_filters = 3  # keeps track of prev conv layer channel , initalized to 3 as this is input #channels
    output_filters = []  # kee[s track of ouput filters of layers (used for route blocks)

    for index, block in enumerate(blocks_[1:]):  # starting from the first real layer
        module = nn.Sequential()  # sequential module as layers have conv + bn + act

        # check type of block , create module and append to modulelist
        # if the layer is a convolutional layer
        if block["type"] == "convolutional":
            activation = block["activation"]
            try:  # not all lauers have batch normalize
                batch_normalize = int(block["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            num_filters = int(block["filters"])
            padding = int(block["pad"])
            kernel_size = int(block["size"])
            stride = int(block["stride"])

            if padding:
                pad = (kernel_size - 1) // 2  # same padding is used overall the network
            else:
                pad = 0

            # Add the convolutional layer to the sequential module
            conv = nn.Conv2d(prev_filters, num_filters, kernel_size, stride, pad, bias=bias)
            module.add_module(f"conv_{index}", conv)

            # Add the Batch Norm layer to the sequential module if exists
            if batch_normalize:
                bn = nn.BatchNorm2d(num_filters)
                module.add_module(f"batch_norm_{index}", bn)

            # Check the activation type.
            # activations are either leakyrelu  or lineear
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module(f"leaky_{index}", activn)

        # if the layer is an upsampling layer
        elif block["type"] == "upsample":
            stride = int(block["stride"])
            upsample = nn.Upsample(scale_factor=2, mode="bilinear",align_corners=True)
            module.add_module(f"upsample_{index}", upsample)

        # if the layer is  a route layer
        elif block["type"] == "route":
            block["layers"] = block["layers"].split(',')  # if two layers are to be concatenated
            # Start  of a route
            start = int(block["layers"][0])
            # end, if there exists one.
            try:
                end = int(block["layers"][1])  # not all routes have two layers to concatenate
            except:
                end = 0
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()  # route will be done by hand in the module
            module.add_module(f"route_{index}", route)
            if end < 0:
                num_filters = output_filters[index + start] + output_filters[index + end]  # when concatenation
            else:
                num_filters = output_filters[index + start]

        # if the layer is a skip connection
        elif block["type"] == "shortcut":
            shortcut = EmptyLayer()  # shortcut will be done by hand in the module
            module.add_module(f"shortcut_{index}", shortcut)

        # If the layer is a detection layer i.e "YOLO"
        elif block["type"] == "yolo":
            masks = block["mask"].split(",")
            mask = [int(mask) for mask in masks]

            anchors = block["anchors"].split(",")
            anchors = [int(anchor) for anchor in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module(f"Detection_{index}", detection)

        module_list.append(module)
        prev_filters = num_filters
        output_filters.append(num_filters)
    return net_info, module_list


# ----------------------------------------------------------------------------
# empty layer class for both the shortcut and the route layers
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


# ----------------------------------------------------------------------------
# a layers that holds anchors of each cell
class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


# ----------------------------------------------------------------------------
# The whole model class
class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, GPU=True):
        modules = self.blocks[1:]
        outputs = {}  # outputs for the route layer or shortcut (skip connection layer)
        first_initalized = 0  # flag to determine that the first scale has been obtained
        for index, module in enumerate(modules):
            module_type = (module["type"])

            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[index](x)  # previous input into the model

            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:  # start>0
                    layers[0] = layers[0] - index

                if len(layers) == 1:  # route from a preceding layer (not necesseceraly the previous one)
                    x = outputs[index + (layers[0])]  # current output is the output of this preceding layer

                else:
                    if (layers[1]) > 0:  # when we concat two layers from different locations in the module
                        layers[1] = layers[
                                        1] - index  # to get a negative value if the layer index is before the current index

                    map1 = outputs[index + layers[0]]  # featrue map of start layer
                    map2 = outputs[index + layers[1]]  # feature map of end layer

                    x = torch.cat((map1, map2), 1)  # concatenated featrue maps along the #channels dimensions

            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[index - 1] + outputs[index + from_]  # output of previous layer concatenated
                # with the desired layer
            elif module_type == 'yolo':

                anchors = self.module_list[index][0].anchors  # get anchors
                # get the input dimensions
                input_dim = int(self.net_info["height"])

                # get the number of classes
                num_classes = int(module["classes"])

                # transform
                x = transform_predictions(x, input_dim, anchors, num_classes, GPU)
                if not first_initalized:
                    detections = x
                    first_initalized = 1

                else:
                    detections = torch.cat((detections, x), 1)

            outputs[index] = x

        try:
            return detections
        except:
            return 0

    def load_weights(self, weightfile):  # loading pretrained weights from serialized binary file
        file = open(weightfile, "rb")
        header = np.fromfile(file, dtype=np.int32, count=5)  # first 5 nummbers are headers
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(file, dtype=np.float32)  # the rest are weights
        ptr = 0
        for i in range(len(self.module_list)):  # iterate module list and load weigghts
            module_type = self.blocks[i + 1]["type"]  # ignoring the first net block in cfg
            if module_type == "convolutional":
                model = self.module_list[i]
                try:  # not all conv layers have bn
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]  # first part of model is conv
                if (batch_normalize):
                    bn = model[1]  # second part is batch nor,

                    # get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    # load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases  # num of biases same as num of weights

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # the following lines make the biases dimensions obtained same as those in model
                    bn_biases = bn_biases.view_as(bn.bias)
                    bn_weights = bn_weights.view_as(bn.weight)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # cooies the data into the model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                else:  # if no batch norm load biases of conv layer
                    # Number of biases
                    num_biases = conv.bias.numel()

                    # load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias)

                    # finally copy the data
                    conv.bias.data.copy_(conv_biases)
                # in both cases weights of conv layer should be loaded
                num_weights = conv.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


# ----------------------------------------------------------------------------
# getting a single image for testing the model output
def get_test_input(imgpath):
    try:
        img = cv2.imread(imgpath)
        img = cv2.resize(img, (416, 416))  # Resize to the input dimension
        img_ = img[:, :, ::-1].transpose((2, 0, 1))  # BGR -> RGB | H X W C -> C X H X W
        img_ = img_[np.newaxis, :, :, :] / 255.0  # Add a channel at 0 (for batch) | Normalise
        img_ = torch.from_numpy(img_).float()  # Convert to float
        img_ = Variable(img_)  # Convert to Variable
        return img_.to("cuda")  # put on the gpu
    except:
        print("wrong image path")


# ----------------------------------------------------------------------------
# just a simple test for model correct output shape 1X10647x85
def test_model_shape(imgpath):
    img = get_test_input(imgpath)
    model = Darknet("yolov3.cfg")
    model.to("cuda")
    model.load_weights("yolov3.weights")
    prediction = model(img, torch.cuda.is_available())
    print(prediction.shape)

