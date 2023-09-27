import open3d as o3d
import numpy as np

class Pose_viz():
    def __init__(self) -> None:

        self.img_width = 640
        self.img_height = 480

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(height=self.img_height, width=self.img_width)

        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0.2, 0.2, 0.2])

        ctr = self.vis.get_view_control()
        init_param = ctr.convert_to_pinhole_camera_parameters()
        rotate_z_rad = np.pi/2
        init_param.extrinsic = np.array([[1,  0,                     0,                     0],
                                         [0,  np.cos(rotate_z_rad), -np.sin(rotate_z_rad),  0],
                                         [0,  np.sin(rotate_z_rad),  np.cos(rotate_z_rad),  0],
                                         [0,  0,                     0,                     1]])
        ctr.convert_from_pinhole_camera_parameters(init_param)
        """
        init_param.extrinsic = np.array([[np.cos(rotate_z_rad), -np.sin(rotate_z_rad),  0,              0],
                                         [np.sin(rotate_z_rad),  np.cos(rotate_z_rad),  0,              0],
                                         [0,                     0,                     1,              0],
                                         [0,                     0,                     0,              1]])
        
        """
        """
        init_param.extrinsic = np.array([[0, -1,  0,  0],
                                         [1,  0,  0,  0],
                                         [0,  0,  1,  0],
                                         [0,  0,  0,  1]])
        """

        self.pose_lines = [
            [0, 2], [0, 5], [2, 1], [2, 3], [4, 5], 
            [5, 6], [2, 7], [5, 8], [9, 10], [11, 12], 
            [12, 14], [11, 13], [14, 16], [16, 22], 
            [16, 20], [16, 18], [18, 20], [13, 15],
            [15, 21], [15, 19],[15, 17],[17, 19],
            [11, 23], [12, 24], [23, 24], [23, 25],
            [24, 26], [25, 27], [26, 28], [27, 29],
            [29, 31], [28, 32], [28, 30], [30, 32]
        ]
        self.pose_line_colors = [[1, 0, 0] for i in range(len(self.pose_lines))]


        self.l_hand_pcd = o3d.geometry.PointCloud()
        self.pose_pcd   = o3d.geometry.PointCloud()
        self.r_hand_pcd = o3d.geometry.PointCloud()
        self.face_pcd   = o3d.geometry.PointCloud()

        self.line_set   = o3d.geometry.LineSet()

        self.null_points = o3d.utility.Vector3dVector([])
        self.null_lines = o3d.utility.Vector2iVector([])

        #mesh_box = o3d.geometry.TriangleMesh.create_box(width=30.0, height=30.0, depth=0.1)
        #self.vis.add_geometry(mesh_box)

        self.l_hand_added   = False
        self.r_hand_added   = False
        self.face_added     = False
        self.pose_added     = False

    def landmarks_to_points(self, data):
        #data = np.array(landmarks.pose_landmarks.landmark)
        xyz = np.array([])

        for point in data:
            p = np.array([point.x, point.y, point.z])
            xyz = np.append(xyz, p)
        xyz = xyz.reshape((len(data), 3))

        points = o3d.utility.Vector3dVector(xyz) 
        #self.pcd.points = o3d.utility.Vector3dVector(xyz)
        return xyz

    def visualize3d(self, landmarks):
        if landmarks.pose_landmarks is not None:
            pose_points = self.landmarks_to_points(np.array(landmarks.pose_landmarks.landmark))
            self.pose_pcd.points = o3d.utility.Vector3dVector(pose_points)
            self.line_set.points = o3d.utility.Vector3dVector(pose_points)
            self.line_set.lines = o3d.utility.Vector2iVector(self.pose_lines)
            self.line_set.colors = o3d.utility.Vector3dVector(self.pose_line_colors)

            self.pose_pcd.paint_uniform_color([0, 1, 0])

            if self.pose_added:
                self.vis.update_geometry(self.pose_pcd)
                self.vis.update_geometry(self.line_set)
            else:
                self.pose_added = True
                self.vis.add_geometry(self.pose_pcd)
                self.vis.add_geometry(self.line_set)
        else:
            self.pose_pcd.points = self.null_points
            self.vis.update_geometry(self.pose_pcd)
            self.line_set.lines = self.null_lines
            self.vis.update_geometry(self.line_set)
            

        if landmarks.right_hand_landmarks is not None:
            r_h_points = self.landmarks_to_points(np.array(landmarks.right_hand_landmarks.landmark))

            r_h_points[:,2] += pose_points[16][2]

            self.r_hand_pcd.points = o3d.utility.Vector3dVector(r_h_points)
            self.r_hand_pcd.paint_uniform_color([0, 1, 1])
            if self.r_hand_added:
                self.vis.update_geometry(self.r_hand_pcd)
            else:
                self.r_hand_added = True
                self.vis.add_geometry(self.r_hand_pcd)               
        else:
            self.r_hand_pcd.points = self.null_points
            self.vis.update_geometry(self.r_hand_pcd)

        if landmarks.left_hand_landmarks is not None:
            l_h_points = self.landmarks_to_points(np.array(landmarks.left_hand_landmarks.landmark))

            l_h_points[:,2] += pose_points[15][2]

            self.l_hand_pcd.points = o3d.utility.Vector3dVector(l_h_points)
            self.l_hand_pcd.paint_uniform_color([0, 1, 1])
            if self.l_hand_added:
                self.vis.update_geometry(self.l_hand_pcd)
            else:
                self.l_hand_added = True
                self.vis.add_geometry(self.l_hand_pcd)  
        else:
            self.l_hand_pcd.points = self.null_points
            self.vis.update_geometry(self.l_hand_pcd)

        if landmarks.face_landmarks is not None:
            face_points = self.landmarks_to_points(np.array(landmarks.face_landmarks.landmark))

            face_points[:,2] += pose_points[0][2]
            
            self.face_pcd.points = o3d.utility.Vector3dVector(face_points)
            self.face_pcd.paint_uniform_color([1, 1, 0])
            if self.face_added:
                self.vis.update_geometry(self.face_pcd)
            else:
                self.face_added = True
                self.vis.add_geometry(self.face_pcd)  
        else:
            self.face_pcd.points = self.null_points
            self.vis.update_geometry(self.face_pcd)


        self.vis.poll_events()
        self.vis.update_renderer()
    
    def destroy(self):
        self.vis.destroy_window()