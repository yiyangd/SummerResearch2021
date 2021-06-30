import pyrealsense2 as rs 
# https://intelrealsense.github.io/librealsense/python_docs/
import numpy as np

class DepthCamera:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


        
        # Start streaming
        self.pipeline.start(config)

    def get_frame(self):
        ### point cloud
        pc = rs.pointcloud()
        points = rs.points()
        ###

        frames = self.pipeline.wait_for_frames() #
        # The alignment utility performs per-pixel geometric transformation 
        # based on the DEPTH Data
        align_to = rs.stream.color
        align = rs.align(align_to)
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        # Get aligned frames
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        ### Points and Vertices
        points = pc.calculate(depth_frame)
        vertices = np.asanyarray(points.get_vertices(dims=2))
        w = depth_frame.get_width()
        image_Points = np.reshape(vertices, (-1,w,3))
        ###

        # each arrived frame is converted to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        if not depth_frame or not color_frame:
            return False, None, None
        return True, depth_image, color_image, image_Points

    def release(self):
        self.pipeline.stop()