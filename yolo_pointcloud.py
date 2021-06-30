# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

from timeit import default_timer as timer
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
#from keras.utils import multi_gpu_model
import tensorflow.compat.v1 as tf1 
tf1.disable_v2_behavior()
import colorsys, os, cv2
import pyrealsense2 as rs
import numpy as np
from realsense_pointcloud import *

import open3d as o3d

point = (400, 300)

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (640, 480), 
        "gpu_num" : 0,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self):
        self.__dict__.update(self._defaults) # set up default values
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image, depth_frame):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        ### Create Lists for labels and all objects' 4 keys
        labels = []
        lefts = []
        rights = []
        tops = []
        bottoms = []
        ###

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            print(label, (left, top), (right, bottom))
            labels.append(label)
            lefts.append(left)
            rights.append(right)
            tops.append(top)
            bottoms.append(bottom)

            cols = right - left + 1
            rows = bottom - top + 1
            depth_matrix = np.arange(cols*rows).reshape((rows,cols))
            for row in range(0, rows):
                for col in range(0, cols):
                    depth_matrix[row][col] = depth_frame[row+top-1][col+left-1]
            print("matrix shape: {0}\n".format(depth_matrix.shape))
            print(depth_matrix)

            ####
            np.savetxt('depth_matrix.csv',depth_matrix,fmt='%d')

            print("depth_matrix saved!")
            #### 

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

           
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
            

        end = timer()
        print(end - start)
        return image, lefts, rights, tops, bottoms, labels

    def close_session(self):
        self.sess.close()



def show_distance(event, x, y, args, params):
    global point
    point = (x, y)

def detect_video(yolo):
    # Initialize Camera Intel Realsense
    dc = DepthCamera()

    # Create mouse event
    cv2.namedWindow("depth frame")
    # Make Point vary as I move the mouse
    cv2.setMouseCallback("depth frame", show_distance) 


    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        ret, depth_frame, color_frame, image_Points = dc.get_frame()
        rgb = color_frame[...,::-1]
        # Show distance for a specific point
        cv2.circle(color_frame, point, 4, (0, 0, 255))

        # Read distance from DEPTH FRAME!!!!!
        # y_th row and x_th column
        distance = depth_frame[point[1], point[0]]

        image = Image.fromarray(color_frame)
        image, lefts, rights, tops, bottoms, labels = yolo.detect_image(image, depth_frame) 
        ### Write point_cloud.ply
        for i in range(0, len(labels)):
            left = lefts[i]
            right = rights[i]
            top = tops[i]
            bottom = bottoms[i]
            vertices_interest = image_Points[top:bottom, left:right, :].reshape(-1,3)
            color_interest = color_frame[top:bottom, left:right, :].reshape(-1,3)
            #### open3D
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices_interest.astype(np.float32)/255)
            pcd.colors = o3d.utility.Vector3dVector(color_interest.astype(np.float32)/255)
            #pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            o3d.visualization.draw_geometries([pcd])
            o3d.io.write_point_cloud(labels[i]+'.ply',pcd)
            print(labels[i]+'.ply saved!')
        ###

        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0

        #cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
        cv2.putText(result, "{}m".format(distance/1000), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        cv2.putText(result,"x: {}, y:{}".format(point[0], point[1]), (point[0], point[1] - 45), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        # the following line doesn't work well
        cv2.putText(depth_frame, "{}m".format(distance/1000), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        #cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #            fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
            # Syntax: 
        cv2.imshow("depth frame", depth_frame)
        #cv2.imshow("RGB frame", color_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()

if __name__ == '__main__':
    detect_video(YOLO())
