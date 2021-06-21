import cv2
import pyrealsense2
from realsense_depth import *

point = (400, 300)

def show_distance(event, x, y, args, params):
    global point
    point = (x, y)

# Initialize Camera Intel Realsense
dc = DepthCamera()

# Create mouse event
cv2.namedWindow("depth frame")
# Make Point vary as I move the mouse
cv2.setMouseCallback("depth frame", show_distance) 

while True:
    ret, depth_frame, color_frame = dc.get_frame()

    # Show distance for a specific point
    cv2.circle(color_frame, point, 4, (0, 0, 255))

    # Read distance from DEPTH FRAME!!!!!
    distance = depth_frame[point[1], point[0]]

    # Syntax: 
    #cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
    cv2.putText(color_frame, "{}m".format(distance/1000), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    
    # the following line doesn't work well
    cv2.putText(depth_frame, "{}m".format(distance/1000), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    
    cv2.imshow("depth frame", depth_frame)
    cv2.imshow("RGB frame", color_frame)

    key = cv2.waitKey(1)
    if key == 27:
        break