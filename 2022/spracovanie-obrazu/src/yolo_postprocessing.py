import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from os import path

def np_sigmoid(x):
    #return expit(x)
    return tf.math.sigmoid(x)

def nonMaxSuppress(boxLoc, score, classPredict, maxBox=20, iouThresh=0.5):
    score = tf.cast(score, dtype=tf.float32)
    
    selected_indexes = tf.image.non_max_suppression(boxLoc, score, maxBox, iou_threshold=iouThresh)
    selected_boxes = tf.gather(boxLoc, selected_indexes)
    selected_score = tf.gather(score, selected_indexes)
    selected_classes = tf.gather(classPredict, selected_indexes)
    
    return selected_boxes, selected_score, selected_classes

def np_yolo_correct_boxes(box_xy, box_wh, input_shape):
    input_shape = input_shape.astype(np.float32)

    #change order
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    #corner coordinates of the boxes
    box_mins = (box_yx - (box_hw / 2.)) / input_shape
    box_maxes = (box_yx + (box_hw / 2.)) / input_shape
    
    boxes = np.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)
    
    return boxes

def np_yolo_head(feats,
                 anchors,
                 num_classes,
                 input_shape,
				 factor):
    num_anchors = 3

    #used anchors
    anchors_tensor = np.reshape(anchors, [1, 1, 1, num_anchors, 2])

    #create grid index system
    grid_shape = (feats.shape[1], feats.shape[2])  # height, width
    grid_y = np.tile(np.reshape(np.arange(0, grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
    grid_x = np.tile(np.reshape(np.arange(0, grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
    grid = np.concatenate([grid_x, grid_y], axis=-1).astype(np.float32)
    
    # (batch_size, 13 13, 3, 85)
    feats = np.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5]).astype(np.float64)

    # decode the raw prediction
    # box_xy is the center of ground truth
    # box_wh is the w and h of gt
    
    #grid box size
    stride = input_shape[::-1] // grid_shape[::-1]

    box_xy = ((np_sigmoid(feats[..., :2]) * factor) - 0.5 * (factor - 1) + grid) * stride
    box_wh = np.exp(feats[..., 2:4]) * anchors_tensor

    box_confidence = np_sigmoid(feats[..., 4:5])
    box_class_probs = np_sigmoid(feats[..., 5:])

    return box_xy, box_wh, box_confidence, box_class_probs

def yolov4_decoder(predictions):
    if len(predictions) == 2:
        #print("Tiny model")
        
        anchors = [[[23, 27], 
                    [37, 58],
                    [81, 82]],
                   [[81, 82],
                    [135, 169],
                    [344, 319]]]
        sigmoid_factor = [1.05, 1.05]
        grid_size = [16, 32]
    elif len(predictions) == 3:
        anchors = [[[12, 16], 
                    [19, 36],
                    [40, 28]],
                   [[36, 75],
                    [76, 55],
                    [72, 146]],
                   [[142, 110],
                    [192, 243],
                    [459, 401]]]
        sigmoid_factor = [1.2, 1.1, 1.05]
        grid_size = [8, 16, 32]
    else:
        raise Exception("Invalid number of output tensors. Either 3 or 2, actual %d" % len(output_tensors))

    num_tensors = len(predictions)
    num_channels = 3
    num_classes = predictions[0].shape[-1] // num_channels - 5
	
    bbox_list = []
    score_list = []
    input_shape = np.array([predictions[0].shape[1] * grid_size[0],
	                        predictions[0].shape[2] * grid_size[0]])

    for idx, tensor in enumerate(predictions):
        grid_shape = np.shape(tensor)[1:3]
        temp_tensor = np.reshape(tensor, [-1, grid_shape[0], grid_shape[1], num_channels, num_classes + 5])
   
        box_xy, box_wh, box_confidence, box_class_probs = np_yolo_head(temp_tensor, anchors[idx], num_classes,
                                                                       input_shape, sigmoid_factor[idx])
        bboxes_xywh = np_yolo_correct_boxes(box_xy, box_wh, input_shape)

        bboxes_xywh = np.reshape(bboxes_xywh, [temp_tensor.shape[0], -1, 4])

        bboxes_scores = box_confidence * box_class_probs
        bboxes_scores = np.reshape(bboxes_scores, [temp_tensor.shape[0], -1, num_classes])

        bbox_list.append(bboxes_xywh)
        score_list.append(bboxes_scores)

    boxes = np.concatenate(bbox_list, axis=1).astype(np.float32)
    box_scores = np.concatenate(score_list, axis=1).astype(np.float32)
    
    background_score = np.expand_dims(np.zeros(box_scores.shape[:-1]), axis=-1)
    box_scores = np.concatenate([background_score, box_scores], axis=-1)
    
    return correctBoxes(boxes, box_scores)


def correctBoxes(boxes, scores):
    #take max score at every box
    scores_max = tf.math.reduce_max(scores, axis=-1)

    #create mask from boxes, where max score is higher that 0.4
    mask = scores_max >= 0.4

    #apply mask => boxes with highest scores
    class_boxes = tf.boolean_mask(boxes, mask)
    pred_conf = tf.boolean_mask(scores, mask)
    box_scores = tf.boolean_mask(scores_max, mask)

    print("Number of boxes with score higher that threshold: {}".format(class_boxes.shape[0]))

    if class_boxes.shape[0] == 0:
        raise Exception("No objects detected.")

    else:
        #get index of the max score at every box => class 
        class_id = np.argmax(pred_conf, axis=-1)
        
        #prunes away boxes, that have high Intersection Over Union
        boxes, scores, classes = nonMaxSuppress(class_boxes, box_scores, class_id)
        print("Number of valid bounding boxes found: {}".format(boxes.shape[0]))

        return boxes, scores, classes


def decoder(predictions):
    def grids(t):
        return t.shape[1]
    predictions.sort(key=grids, reverse=True)

    return yolov4_decoder(predictions)
