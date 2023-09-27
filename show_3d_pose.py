import numpy as np
import matplotlib.pyplot as plt
from utils import read_rotation_translation
import time
plt.style.use("seaborn-v0_8")
import open3d as o3d
import cv2 as cv

def read_keypoints(filename, keypoints):
    fin = open(filename, "r")

    kpts = []
    while True:
        line = fin.readline()
        if line == "":
            break

        line = line.split()
        line = [float(s) for s in line]

        line = np.reshape(line, (len(keypoints), -1))
        kpts.append(line)

    kpts = np.array(kpts)
    return kpts


def visualize_3d(p3ds):
    """Now visualize in 3D"""
    torso = [[0, 1], [1, 7], [7, 6], [6, 0]]
    armr = [[1, 3], [3, 5]]
    arml = [[0, 2], [2, 4]]
    legr = [[6, 8], [8, 10]]
    legl = [[7, 9], [9, 11]]
    body = [torso, arml, armr, legr, legl]
    colors = ["red", "blue", "green", "black", "orange"]

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for framenum, kpts3d in enumerate(p3ds):
        if framenum % 2 == 0:
            continue  # skip every 2nd frame
        for bodypart, part_color in zip(body, colors):
            for _c in bodypart:
                ax.plot(
                    xs=[kpts3d[_c[0], 0], kpts3d[_c[1], 0]],
                    ys=[kpts3d[_c[0], 1], kpts3d[_c[1], 1]],
                    zs=[kpts3d[_c[0], 2], kpts3d[_c[1], 2]],
                    linewidth=4,
                    c=part_color,
                )

        # uncomment these if you want scatter plot of keypoints and their indices.
        # for i in range(12):
        #    ax.text(kpts3d[i,0], kpts3d[i,1], kpts3d[i,2], str(i))
        #    ax.scatter(xs = kpts3d[i:i+1,0], ys = kpts3d[i:i+1,1], zs = kpts3d[i:i+1,2])

        # ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.set_xlim3d(-100, 100)
        ax.set_xlabel("x")
        ax.set_ylim3d(-100, 100)
        ax.set_ylabel("y")
        ax.set_zlim3d(-100, 100)
        ax.set_zlabel("z")
        plt.pause(0.1)
        ax.cla()


def visualize_open3d(kpts_3d_face, kpts_3d_left_hand, kpts_3d_right_hand, kpts_3d_body, pose_list):

    # input video stream
    cap0 = cv.VideoCapture('media/camera_0_mp.avi')
    cap1 = cv.VideoCapture('media/camera_1_mp.avi')

    caps = [cap0, cap1]

    win_width = 1280
    win_height = 720

    camera_resolution = [720, 1280]
    scale_percent = 60 # percent of original size
    width_resizes = int(camera_resolution[1] * scale_percent / 100)
    height_resized = int(camera_resolution[0] * scale_percent / 100)


    # set camera resolution if using webcam to 1280x720.
    for cap in caps:
        cap.set(3, camera_resolution[1])
        cap.set(4, camera_resolution[0])


    vis = o3d.visualization.Visualizer()
    vis.create_window(height=896, width=1139)
    

    #cv.projectPoints()
    #P1 = get_projection_matrix(1)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    opt.point_size = 0.5
    
    
    hand_lines = np.array(
        [
            [0, 1], [1, 2], [2, 3], [3, 4], 
            [0, 5], [0, 17], [5, 9], [9, 13], [13, 17],
            [5, 6], [6, 7], [7, 8], [9, 10], [10, 11], 
            [11, 12], [13, 14], [14, 15], [15, 16],
            [17, 18], [18, 19], [19, 20] 
        ]
    )

    #if "upper_body" in pose_list:
    #    pose_lines = np.array([[11, 12], [11, 13], [12, 14], [13, 15], [14, 16]]) - 11
    pose_lines = [
        [0, 2], [0, 5], [2, 1], [2, 3],
        [4, 5], [5, 6], [2, 7], [5, 8],
        [9, 10], [11, 12], [12, 14], [11, 13],
        [14, 16], [16, 22], [16, 20], [16, 18],
        [18, 20], [13, 15], [15, 21], [15, 19],
        [15, 17], [17, 19], [11, 23], [12, 24],
        [23, 24], [23, 25], [24, 26], [25, 27],
        [26, 28], [27, 29], [29, 31], [28, 32],
        [28, 30], [30, 32]]

    pose_lines = [
        [0, 2], [0, 5], [2, 1], [2, 3],
        [4, 5], [5, 6], [2, 7], [5, 8],
        [9, 10], [11, 12], [12, 14], [11, 13],
        [14, 16], [16, 22], [16, 20], [16, 18],
        [18, 20], [13, 15], [15, 21], [15, 19],
        [15, 17], [17, 19], [11, 23], [12, 24],
        [23, 24]]


    # [[left,           right],
    #  [from_height,    to_height],
    #  [from_dist,      to_dist]]
    ROI = np.array([[-120,    120],
                    [ -80,   122],
                    [ -50,   -400]])

    # Create box


    #Box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(points))


    """
    pose_lines = [[0, 1] , [1, 7], [7, 6], [6, 0], 
                  [1, 3], [3, 5], [0, 2], [2, 4], 
                  [6, 8], [8, 10], [7, 9], [9, 11]]
    """


    box_line_set   = o3d.geometry.LineSet()
    box_points = boundPolyCreate(ROI)
    #box_lines = [[4, 5], [1, 5], [0,2], [4, 6], [5, 7], [2, 6], [6, 7]]
    box_lines = [[2, 3], [2, 6], [6, 7], [3, 7], [5, 7], [6, 4], [1, 5], [1, 3], [4, 5]]

    box_lines_colors = [[0.5, 0.5, 0.5] for i in range(len(box_lines))]

    pose_line_colors = [[1, 0, 0] for i in range(len(pose_lines))]
    hand_lines_colors_l = [[0, 1, 0] for i in range(len(hand_lines))]
    hand_lines_colors_r = [[0, 1, 1] for i in range(len(hand_lines))]

    box_line_set.points = o3d.utility.Vector3dVector(box_points)
    box_line_set.lines = o3d.utility.Vector2iVector(box_lines)
    box_line_set.colors = o3d.utility.Vector3dVector(box_lines_colors)


    pose_pcd = o3d.geometry.PointCloud()
    left_hand_pcd = o3d.geometry.PointCloud()
    right_hand_pcd = o3d.geometry.PointCloud()
    face_pcd = o3d.geometry.PointCloud()


    body_line_set = o3d.geometry.LineSet()
    body_line_set.colors = o3d.utility.Vector3dVector(pose_line_colors)
    body_line_set.lines = o3d.utility.Vector2iVector(pose_lines)
    
    hand_line_set_l = o3d.geometry.LineSet()
    hand_line_set_l.colors = o3d.utility.Vector3dVector(hand_lines_colors_l)
    hand_line_set_l.lines = o3d.utility.Vector2iVector(hand_lines)
    
    hand_line_set_r = o3d.geometry.LineSet()
    hand_line_set_r.colors = o3d.utility.Vector3dVector(hand_lines_colors_r)
    hand_line_set_r.lines = o3d.utility.Vector2iVector(hand_lines)

    first = True

    while True:
        cap0 = cv.VideoCapture('media/camera_0_mp.avi')
        cap1 = cv.VideoCapture('media/camera_1_mp.avi')
        
        for framenum, _ in enumerate(kpts_3d_body):
            time.sleep(1/30)
            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()

            pose_pcd.points = o3d.utility.Vector3dVector(kpts_3d_body[framenum][:25])
            left_hand_pcd.points = o3d.utility.Vector3dVector(kpts_3d_left_hand[framenum])
            right_hand_pcd.points = o3d.utility.Vector3dVector(kpts_3d_right_hand[framenum])
            face_pcd.points = o3d.utility.Vector3dVector(kpts_3d_face[framenum])

            body_line_set.points = o3d.utility.Vector3dVector(kpts_3d_body[framenum][:25])
            hand_line_set_l.points = o3d.utility.Vector3dVector(kpts_3d_left_hand[framenum])
            hand_line_set_r.points = o3d.utility.Vector3dVector(kpts_3d_right_hand[framenum])
            
            pose_pcd.paint_uniform_color([0, 1, 0])
            left_hand_pcd.paint_uniform_color([0, 0.5, 0.7])
            right_hand_pcd.paint_uniform_color([1, 0.5, 0.7])
            face_pcd.paint_uniform_color([1, 1, 0.5])

            if framenum == 0 and first:
        
                camera_1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=[0, 0, 0])
                camera_2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=[0, 0, 0])
                rot, trans = read_rotation_translation(1)
                camera_2.translate(trans)
                camera_2.rotate(rot)

                vis.add_geometry(camera_1)
                vis.add_geometry(camera_2)
                vis.add_geometry(box_line_set)

                vis.add_geometry(left_hand_pcd)
                vis.add_geometry(right_hand_pcd)
                vis.add_geometry(face_pcd)
                vis.add_geometry(pose_pcd)

                vis.add_geometry(body_line_set)
                vis.add_geometry(hand_line_set_l)
                vis.add_geometry(hand_line_set_r)
                

                ctr = vis.get_view_control()
                parameters = o3d.io.read_pinhole_camera_parameters("vis_open3d/vis_camera.json")
                ctr.convert_from_pinhole_camera_parameters(parameters)

                vis.update_geometry(pose_pcd)
                vis.update_geometry(left_hand_pcd)
                vis.update_geometry(right_hand_pcd)
                vis.update_geometry(face_pcd)
                vis.update_geometry(body_line_set)
                vis.update_geometry(hand_line_set_l)
                vis.update_geometry(hand_line_set_r)

                vis.poll_events()
                vis.update_renderer()
            else:
      

                first = False
                vis.update_geometry(pose_pcd)
                vis.update_geometry(left_hand_pcd)
                vis.update_geometry(right_hand_pcd)
                vis.update_geometry(face_pcd)

                vis.update_geometry(body_line_set)
                vis.update_geometry(hand_line_set_l)
                vis.update_geometry(hand_line_set_r)

            

            resized0 = cv.resize(frame0, (width_resizes, height_resized), interpolation = cv.INTER_AREA)
            resized1 = cv.resize(frame1, (width_resizes, height_resized), interpolation = cv.INTER_AREA)
            cv.imshow("cam0", resized0)
            cv.imshow("cam1", resized1)                       
            event = vis.poll_events()
            vis.update_renderer()

            if not event:
                break

        if not event:
            vis.destroy_window()
            break

        


# ROI bounding box 
def boundPolyCreate(ROI) -> list:        # Create bounding box points
    x, y, z = np.column_stack((ROI[..., 0], ROI[..., 1]))

    return [[x[0], y[0], -z[0]],
            [x[1], y[0], -z[0]],
            [x[0], y[1], -z[0]],
            [x[1], y[1], -z[0]],
            [x[0], y[0], -z[1]],
            [x[1], y[0], -z[1]],
            [x[0], y[1], -z[1]],
            [x[1], y[1], -z[1]]]

if __name__ == "__main__":
    
    FACE_LANDMARKS = 468
    POSE_LANDMARKS = 33
    HAND_LANDMARKS = 21

    face_keypoints = list(range(FACE_LANDMARKS))
    hand_keypoints = list(range(HAND_LANDMARKS))
    pose_keypoints = list(range(POSE_LANDMARKS))

    POSE = ["upper_body"]
    
    """
    pose_landmarks = {
            "head": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "upper_body": [11, 12, 13, 14, 15, 16],
            "left_hand": [17, 19, 21],
            "right_hand": [18, 20, 22],
            "lower_body": [23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
        }
    pose_keypoints = []

    for body_part in pose_landmarks:
        if body_part in POSE:
            pose_keypoints.extend(pose_landmarks[body_part])
    """

    kpts_3d_face = read_keypoints("key_points/kpts_3d_face.dat", face_keypoints)
    kpts_3d_left_hand = read_keypoints("key_points/kpts_3d_left_hand.dat", hand_keypoints) 
    kpts_3d_right_hand = read_keypoints("key_points/kpts_3d_right_hand.dat", hand_keypoints)
    kpts_3d_body = read_keypoints("key_points/kpts_3d_body.dat", pose_keypoints)

    # visualize_3d(p3ds)
    visualize_open3d(kpts_3d_face, kpts_3d_left_hand, kpts_3d_right_hand, kpts_3d_body, POSE)
