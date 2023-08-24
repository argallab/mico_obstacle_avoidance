import numpy as np
import cv2 as cv
import kabsch_helpers
import calibration_kabsch
import tf

np.set_printoptions(suppress = True) # suppress scientific notation
CHESS_HEIGHT = 7 #7x9 board
CHESS_WIDTH = 9
CHESS_SIZE = 0.0197 #1.97 cm 
NEEDED_DEPTHS = 30 # how many valid depths we need

# Realsense D435 hardcoded transform from optical to color frame
# redefinition of axes from optical to camera
# rosrun tf tf_echo optical color
OPTICAL_TO_COLOR = calibration_kabsch.Transformation(tf.transformations.quaternion_matrix([-0.5, 0.5, -0.5, 0.5])[:3,:3], np.array([0,0,0]))
COLOR_TO_OPTICAL = calibration_kabsch.Transformation(tf.transformations.quaternion_matrix([0.5, -0.5, 0.5, 0.5])[:3,:3], np.array([0,0,0]))

# rosrun tf tf_echo camera_aligned_depth_to_color_frame camera_link
COLOR_TO_CAMERA = calibration_kabsch.Transformation(tf.transformations.quaternion_matrix([0.005, -0.000, -0.001, 1.000])[:3,:3], np.array([0.000, -0.015, -0.000]))

# minimal transform
CAMERA_TO_COLOR = calibration_kabsch.Transformation(tf.transformations.quaternion_matrix([-0.00463353516534, 0.000189657497685, 0.000982505269349, 0.999988734722])[:3,:3], np.array([-0.000289856398012,0.0147355077788,0.0000863225141075]))


class intrinsics:
    def __init__(self):
        self.ppx = None
        self.ppy = None
        self.fx = None
        self.fy = None

# only for running this script from main
def get_images_pyrealsense():
    import pyrealsense2 as rs

    pipe = rs.pipeline()
    cfg = rs.config()
    selected_devices = []                     # Store connected device(s)
    for d in rs.context().devices:
        selected_devices.append(d)
    if not selected_devices:
        print("No RealSense device is connected!")
        rgb_sensor = depth_sensor = None
    else:
        for device in selected_devices:                         
            for s in device.sensors:                              # Show available sensors in each device
                if s.get_info(rs.camera_info.name) == 'RGB Camera':
                    rgb_sensor = s                                # Set RGB sensor
                if s.get_info(rs.camera_info.name) == 'Stereo Module':
                    depth_sensor = s                              # Set Depth sensor

        profile = pipe.start(cfg)                                 # Configure and start the pipeline
        for _ in range(10):                                       # Skip first frames to give syncer and auto-exposure time to adjust
            frameset = pipe.wait_for_frames()

        frameset = pipe.wait_for_frames()  # get frames

        align = rs.align(rs.stream.color) # align to color stream
        frameset = align.process(frameset)
        color_frame = frameset.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_frame = frameset.get_depth_frame()
        depth_frame = kabsch_helpers.post_process_depth_frame(depth_frame) # process depth frame
        depth_image = np.asanyarray(depth_frame.get_data())

        color_intrinsics = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()

        return get_chess_corners(color_image, depth_image, color_intrinsics)

# call this from the ros node
def get_chess_corners(color_image, depth_image, color_intrinsics):
    chessboard_found, corners = cv.findChessboardCorners(color_image, (CHESS_HEIGHT, CHESS_WIDTH))
    if chessboard_found:
        return construct_3d_pts(corners, color_image, depth_image, color_intrinsics)
    else:
        raise Exception("Did not find chessboard corners")


def get_depth(depth_image, x, y):
    # indexing into the data returns depth in millimeters (we divide by 1000 to return in m)
    return depth_image[int(round(y)), int(round(x))]/1000.

# helper function for constructing 3d points from 2d chess corner points
def get_camera_pts(corners_2d, depth_frame, color_frame, color_intrinsics, aruco = False):
    points_3d = np.zeros((len(corners_2d), 3)) # shape should be (63, 3)
    valid_points = [False] * len(corners_2d)
    for i in range(len(corners_2d)):
        if aruco:
            corner = corners_2d[i]
        else:
            corner = corners_2d[i][0]
        # depth = kabsch_helpers.get_depth_at_pixel(depth_frame, corner[0], corner[1])
        depth = get_depth(depth_frame, corner[0], corner[1])

        if depth != 0.0: # for all valid points
            valid_points[i] = True
            X, Y, Z = kabsch_helpers.convert_depth_pixel_to_metric_coordinate(depth, corner[0], corner[1], color_intrinsics)
            points_3d[i][0] = X
            points_3d[i][1] = Y
            points_3d[i][2] = Z
    return(points_3d, valid_points)

def construct_3d_pts(corners, color_image, depth_image, color_intrinsics):
    # refine corners (from https://docs.opencv.org/3.4/dd/d92/tutorial_corner_subpixels.html)
    gray_image = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)
    winSize = (5, 5)
    zeroZone = (-1, -1)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 40, 0.001)
    corners = cv.cornerSubPix(gray_image, corners, winSize, zeroZone, criteria)

    corners_3d, valid_points = get_camera_pts(corners, depth_image, color_image, color_intrinsics) # get 3d corner pts
    valid_points = np.array(valid_points, dtype=bool) # mask to only pts with valid depths
    if len(valid_points[valid_points]) >= NEEDED_DEPTHS:
        return post_processing(corners_3d, valid_points, color_image, color_intrinsics)
    else:
        raise Exception("Did not have enough valid depths")

def get_chess_pts(chessboard_height, chessboard_width, chessboard_size, flip = False, verbose = False):
    # from https://docs.opencv.org/4.x/d7/d53/tutorial_py_pose.html
    # create chessboard corner points in chessboard coord frame (units = m)
    # need to make sure in the same order as points returned by findChessboardCorners
    objp = np.zeros((chessboard_height*chessboard_width,3), np.float32)
    m = np.mgrid[1:chessboard_height+1, 1:chessboard_width+1]
    m[0] = m[0,::-1] * -1 # reverse y axis and make negative
    objp[:,:2] = m.T.reshape(-1,2)
    objp[:, [0,1]] = objp[:, [1,0]]
    if flip: objp = np.flipud(objp)
    if verbose: print(objp)
    return objp * chessboard_size

def post_processing(corners_3d, valid_points, color_image, color_intrinsics):
    # determine if first corner is bottom left (vs top right)
    first = corners_3d[0]
    last = corners_3d[-1]
    flip = first[1] > last[1] # y val of bottom corner greater than top corner

    chess = get_chess_pts(CHESS_HEIGHT, CHESS_WIDTH, CHESS_SIZE, verbose=False, flip=flip)[valid_points].T
    camera = corners_3d[valid_points].T

    # transform camera points to camera_link (color) frame first
    color_camera = OPTICAL_TO_COLOR.apply_transformation(camera)
    camera_link_camera = COLOR_TO_CAMERA.apply_transformation(color_camera)
    [rmat, tvec, rmsd] = calibration_kabsch.calculate_transformation_kabsch(camera_link_camera, chess)

    # debugging_output(rmat, tvec, rmsd, camera, chess)
    if rmsd < 0.1:
        return (rmat, tvec)
    else:
        raise Exception("Kabsch rmsd was too big (> 0.1)")

def debugging_output(rmat, tvec, rmsd, camera, chess):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d, Axes3D

    camera_to_chess = calibration_kabsch.Transformation(rmat, tvec)
    chess_to_camera = camera_to_chess.inverse()

    chess_in_camera = chess_to_camera.apply_transformation(chess)

    camera_in_chess = camera_to_chess.apply_transformation(camera)

    # print(chess)
    # print(camera_in_chess)
    # print(rmsd)

    #TRANSFORMATION PLOTTING FOR DEBUGGING #####       
    fig = plt.figure()
    ax = Axes3D(fig)
    # ax.scatter(valid_camp[0], valid_camp[1], valid_camp[2], c='green')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.view_init(elev=-90, azim=-90)

    ax.scatter(chess[0], chess[1], chess[2], c='purple') 
    ax.scatter(camera[0], camera[1], camera[2], c='green')

    # ax.scatter(camera_in_chess[0], camera_in_chess[1], camera_in_chess[2], c='orange')
    plt.show()
   
    # print_camera_dists(camera.T)


def print_camera_dists(points, axis=None):
    if not axis: # use Euclidian distance
        print("euclidian along y axis")
        print(len(points))
        for i in range(1, len(points)):
            d = abs(dist(points[i-1], points[i]))
            # print(d)
            if d < 0.05 and d-0.0197 > 0.001: print(d-0.0197)  # print if diff > 1 mm
            # if d < 0.05: print(d)
        
        # points = np.reshape(points, (9,7,3))
        # points = np.transpose(points, (1,0,2))
        # points = np.reshape(points, (-1, 3))

        # print("euclidian along x axis")
        # for i in range(1, len(points)):
        #     d = abs(dist(points[i-1], points[i]))
        #     if d < 0.05 and d-0.0197 > 0.001: print(d-0.0197)
        #     # if d < 0.05: print(d)
# 3d euclidian distance
def dist(p1, p2):
    import math
    return math.sqrt((p2[1]-p1[1])**2 + (p2[0]-p1[0])**2 + (p2[2]-p1[2])** 2)

if __name__ == '__main__':
    # pyrealsense get images
    results = get_images_pyrealsense()
    if results:
        rmat, tvec = results
        print(rmat)
        print(tvec)
        

