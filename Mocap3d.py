import cv2 as cv
import mediapipe as mp
import numpy as np
from utils import DLT, get_projection_matrix, write_keypoints_to_disk


class Mocap3D:
    def __init__(self, pose=["head", "upper_body"], hands=False, face=False) -> None:
        self.capture_hands = hands
        self.capture_face = face

        self.fps = 30
        self.min_detection_conf_ = 0.2
        self.min_tracking_conf_ = 0.15
        self.model_complexity_ = 2  # 0, 1 or 2

        FACE_LANDMARKS = 468
        HAND_LANDMARKS = 21
        POSE_LANDMARKS = 33

        self.frame_shape = [720, 1280]
        self.pose_landmark_list = pose
        self.pose_keypoints = []

        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # create holistic keypoints detector objects.
        self.holistic0 = self.mp_holistic.Holistic(
            min_detection_confidence=self.min_detection_conf_,
            min_tracking_confidence=self.min_tracking_conf_,
            model_complexity=self.model_complexity_,
        )
        self.holistic1 = self.mp_holistic.Holistic(
            min_detection_confidence=self.min_detection_conf_,
            min_tracking_confidence=self.min_tracking_conf_,
            model_complexity=self.model_complexity_,
        )

        self.no_detection_max = 5

        self.no_detection_counter_0_pose = 0
        self.no_detection_counter_1_pose = 0
        self.no_detection_counter_0_l_hand = 0
        self.no_detection_counter_1_l_hand = 0
        self.no_detection_counter_0_r_hand = 0
        self.no_detection_counter_1_r_hand = 0
        self.no_detection_counter_0_face = 0
        self.no_detection_counter_1_face = 0


        scale_percent = 60 # percent of original size
        self.width_resizes = int(self.frame_shape[1] * scale_percent / 100)
        self.height_resized = int(self.frame_shape[0] * scale_percent / 100)

        self.video_writer0_mp = cv.VideoWriter('media/camera_0_mp.avi', 
                                cv.VideoWriter_fourcc(*'MJPG'),
                                self.fps, (self.frame_shape[1], self.frame_shape[0]))

        self.video_writer1_mp = cv.VideoWriter('media/camera_1_mp.avi', 
                                cv.VideoWriter_fourcc(*'MJPG'),
                                self.fps, (self.frame_shape[1], self.frame_shape[0]))


        self.ORANGE_COLOR = (0, 102, 255)
        self.PINK_COLOR = (255, 0, 255)
        self.YELLOW_COLOR = (0, 255, 255)
        self.BLACK_COLOR = (0, 0, 0)
        self.WHITE_COLOR = (255, 255, 255)
        self.GREEN_COLOR = (0, 255, 0)


        self.pose_lines_pairs = {
            "head": [
                [0, 2],
                [0, 5],
                [2, 1],
                [2, 3],
                [4, 5],
                [5, 6],
                [2, 7],
                [5, 8],
                [9, 10],
            ],
            "upper_body": [[11, 12], [12, 14], [11, 13], [14, 16], [13, 15]],
            "right_hand": [[16, 22], [16, 20], [16, 18], [18, 20]],
            "left_hand": [[15, 21], [15, 19], [15, 17], [17, 19]],
            "lower_body": [
                [11, 23],
                [12, 24],
                [23, 24],
                [23, 25],
                [24, 26],
                [25, 27],
                [26, 28],
                [27, 29],
                [29, 31],
                [28, 32],
                [28, 30],
                [30, 32],
            ],
        }

        self.pose_landmarks = {
            "head": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "upper_body": [11, 12, 13, 14, 15, 16],
            "left_hand": [17, 19, 21],
            "right_hand": [18, 20, 22],
            "lower_body": [23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
        }

        """
        for body_part in self.pose_landmarks:
            if body_part in self.pose_landmark_list:
                self.pose_keypoints.extend(self.pose_landmarks[body_part])
        """

        self.pose_keypoints = list(range(POSE_LANDMARKS))
        self.face_keypoints = list(range(FACE_LANDMARKS))
        self.hand_keypoints = list(range(HAND_LANDMARKS))


        self.pose_keypoints_0_history = [[-1, -1]] * len(self.pose_keypoints)
        self.face_keypoints_0_history = [[-1, -1]] * len(self.face_keypoints)
        self.l_hand_keypoints_0_history = [[-1, -1]] * len(self.hand_keypoints)
        self.r_hand_keypoints_0_history = [[-1, -1]] * len(self.hand_keypoints)

        self.pose_keypoints_1_history = [[-1, -1]] * len(self.pose_keypoints)
        self.face_keypoints_1_history = [[-1, -1]] * len(self.face_keypoints)
        self.l_hand_keypoints_1_history = [[-1, -1]] * len(self.hand_keypoints)
        self.r_hand_keypoints_1_history = [[-1, -1]] * len(self.hand_keypoints)


    def run_mp(self, input_stream1, input_stream2, P0, P1, record_vid=True, detect=True):
        # input video stream
        cap0 = cv.VideoCapture(input_stream1)
        cap1 = cv.VideoCapture(input_stream2)


        caps = [cap0, cap1]

        # set camera resolution if using webcam to 1280x720. Any bigger will cause some lag for hand detection
        for cap in caps:
            cap.set(3, self.frame_shape[1])
            cap.set(4, self.frame_shape[0])

        # containers for detected keypoints for each camera. These are filled at each frame.
        # This will run you into memory issue if you run the program without stop
        kpts_cam0_face = []
        kpts_cam0_left_hand = []
        kpts_cam0_right_hand = []
        kpts_cam0_body = []

        kpts_cam1_face = []
        kpts_cam1_left_hand = []
        kpts_cam1_right_hand = []
        kpts_cam1_body = []

        kpts_3d_face = []
        kpts_3d_left_hand = []
        kpts_3d_right_hand = []
        kpts_3d_body = []


        while True:
            # read frames from stream
            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()

            if not ret0 or not ret1:
                break

            # crop to 720x720
            #frame0 = self.crop_frame(frame0)
            #frame1 = self.crop_frame(frame1)


            frame0, results0 = self.detect_keypoints(frame0, self.holistic0)
            frame1, results1 = self.detect_keypoints(frame1, self.holistic1)

            """
            face_keypoints_0 = self.keypoints_xy(results0, frame0, 0, body_part="face")
            face_keypoints_1 = self.keypoints_xy(results1, frame1, 1, body_part="face")
            left_hand_keypoints_0 = self.keypoints_xy(results0, frame0, 0, body_part="left_hand")
            left_hand_keypoints_1 = self.keypoints_xy(results1, frame1, 1, body_part="left_hand")
            right_hand_keypoints_0 = self.keypoints_xy(results0, frame0, 0, body_part="right_hand")
            right_hand_keypoints_1 = self.keypoints_xy(results1, frame1, 1, body_part="right_hand")
            pose_keypoints_0 = self.keypoints_xy(results0, frame0, 0,body_part="pose")
            pose_keypoints_1 = self.keypoints_xy(results1, frame1, 1,body_part="pose")
            """
            results = (results0, results1)
            frames = (frame0, frame1)

            face_keypoints_0, face_keypoints_1              = self.stereo_keypoints_xy(results, frames, body_part="face")
            left_hand_keypoints_0, left_hand_keypoints_1    = self.stereo_keypoints_xy(results, frames, body_part="left_hand")
            right_hand_keypoints_0, right_hand_keypoints_1  = self.stereo_keypoints_xy(results, frames, body_part="right_hand")
            pose_keypoints_0, pose_keypoints_1              = self.stereo_keypoints_xy(results, frames, body_part="pose")


            self.plot_keypoints(frame0, face_keypoints_0, color=self.ORANGE_COLOR, radius=2)
            self.plot_keypoints(frame1, face_keypoints_1, color=self.ORANGE_COLOR, radius=2)
            self.plot_keypoints(frame0, left_hand_keypoints_0, color=self.PINK_COLOR)
            self.plot_keypoints(frame1, left_hand_keypoints_1, color=self.PINK_COLOR)
            self.plot_keypoints(frame0, right_hand_keypoints_0, color=self.YELLOW_COLOR)
            self.plot_keypoints(frame1, right_hand_keypoints_1, color=self.YELLOW_COLOR)

            """
            pose_keypoints_0_head = []
            pose_keypoints_0_upper_body = []
            pose_keypoints_0_left_hand = []
            pose_keypoints_0_right_hand = []
            pose_keypoints_0_lower_body = []

            pose_keypoints_1_head = []
            pose_keypoints_1_upper_body = []
            pose_keypoints_1_left_hand = []
            pose_keypoints_1_right_hand = []
            pose_keypoints_1_lower_body = []

            for land_mark_name in self.pose_landmarks.keys():
                if land_mark_name in self.pose_landmark_list:
                    if land_mark_name == "head":
                        for i in self.pose_landmarks[land_mark_name]:
                            pose_keypoints_0_head.append(pose_keypoints_0[i])
                            pose_keypoints_1_head.append(pose_keypoints_1[i])
                                
                    if land_mark_name == "upper_body":
                        for i , body_point in enumerate(self.pose_landmarks[land_mark_name]):
                            pose_keypoints_0_upper_body.append(pose_keypoints_0[i])                       
                            pose_keypoints_1_upper_body.append(pose_keypoints_1[i])   

                    if land_mark_name == "left_hand":
                        for i in self.pose_landmarks[land_mark_name]:
                            pose_keypoints_0_left_hand.append(pose_keypoints_0[i])
                            pose_keypoints_1_left_hand.append(pose_keypoints_1[i])
                                
                    if land_mark_name == "right_hand":
                        for i in self.pose_landmarks[land_mark_name]:
                            pose_keypoints_0_right_hand.append(pose_keypoints_0[i])                       
                            pose_keypoints_1_right_hand.append(pose_keypoints_1[i])   

                    if land_mark_name == "lower_body":
                        for i, body_point in enumerate(self.pose_landmarks[land_mark_name]):
                            pose_keypoints_0_lower_body.append(pose_keypoints_0[i])                       
                            pose_keypoints_1_lower_body.append(pose_keypoints_1[i]) 

                
            self.plot_keypoints(
                frame0, pose_keypoints_0_head, color=self.BLACK_COLOR, radius=5)
            self.plot_keypoints(
                frame1, pose_keypoints_1_head, color=self.BLACK_COLOR, radius=5)
            self.plot_keypoints(
                frame0, pose_keypoints_0_upper_body, color=self.WHITE_COLOR, radius=5)
            self.plot_keypoints(
                frame1, pose_keypoints_1_upper_body, color=self.WHITE_COLOR, radius=5)
            self.plot_keypoints(
                frame0, pose_keypoints_0_left_hand, color=self.GREEN_COLOR, radius=5)
            self.plot_keypoints(
                frame1, pose_keypoints_1_left_hand, color=self.GREEN_COLOR, radius=5)
            self.plot_keypoints(
                frame0, pose_keypoints_0_right_hand, color=self.GREEN_COLOR, radius=5)
            self.plot_keypoints(
                frame1, pose_keypoints_1_right_hand, color=self.GREEN_COLOR, radius=5)
            self.plot_keypoints(
                frame0, pose_keypoints_0_lower_body, color=self.WHITE_COLOR, radius=5)
            self.plot_keypoints(
                frame1, pose_keypoints_1_lower_body, color=self.WHITE_COLOR, radius=5)
            """
            self.plot_keypoints(
                frame0, pose_keypoints_0, color=self.BLACK_COLOR, radius=5)                
            self.plot_keypoints(
                frame1, pose_keypoints_1, color=self.BLACK_COLOR, radius=5)
            

            # this will keep keypoints of this frame in memory
            kpts_cam0_face.append(face_keypoints_0)
            kpts_cam1_face.append(face_keypoints_1)
            kpts_cam0_left_hand.append(left_hand_keypoints_0)
            kpts_cam1_left_hand.append(left_hand_keypoints_1)
            kpts_cam0_right_hand.append(right_hand_keypoints_0)
            kpts_cam1_right_hand.append(right_hand_keypoints_1)
            kpts_cam0_body.append(pose_keypoints_0)
            kpts_cam1_body.append(pose_keypoints_1)


            kpts_3d_face.append(self.calc_3D_points(face_keypoints_0, face_keypoints_1, self.face_keypoints))
            kpts_3d_left_hand.append(self.calc_3D_points(left_hand_keypoints_0, left_hand_keypoints_1, self.hand_keypoints))
            kpts_3d_right_hand.append(self.calc_3D_points(right_hand_keypoints_0, right_hand_keypoints_1, self.hand_keypoints))
            kpts_3d_body.append(self.calc_3D_points(pose_keypoints_0, pose_keypoints_1, self.pose_keypoints))            


            self.video_writer0_mp.write(frame0)
            self.video_writer1_mp.write(frame1)

            resized0 = cv.resize(frame0, (self.width_resizes, self.height_resized), interpolation = cv.INTER_AREA)
            resized1 = cv.resize(frame1, (self.width_resizes, self.height_resized), interpolation = cv.INTER_AREA)
            cv.imshow("cam0", resized0)
            cv.imshow("cam1", resized1)

            k = cv.waitKey(1)
            if k & 0xFF == 27:
                break  # 27 is ESC key.

        for cap in caps:
            cap.release()
        self.video_writer0_mp.release()
        self.video_writer1_mp.release()
        cv.destroyAllWindows()

        mocap_data = (
            np.array(kpts_3d_face), np.array(kpts_3d_left_hand), np.array(kpts_3d_right_hand), np.array(kpts_3d_body)
        )
        return mocap_data

    def plot_pose_lines(self, frame):
        pass
        # cv.line(frame, start_point, end_point, color, thickness)

    def calc_3D_points(self, keypoints0, keypoints1, keypoints):
        # calculate 3d position
        frame_p3ds = []
        for uv1, uv2 in zip(keypoints0, keypoints1):
            if uv1[0] == -1 or uv2[0] == -1:
                _p3d = [-1, -1, -1]
            else:
                _p3d = DLT(P0, P1, uv1, uv2)  # calculate 3d position of keypoint
            frame_p3ds.append(_p3d)

        frame_p3ds = np.array(frame_p3ds).reshape((len(keypoints), 3))
        return frame_p3ds

    def stereo_keypoints_xy(self, mp_detections, frames, body_part="face"):
        frame_0, frame_1 = frames
        landmarks_0, landmarks_1 = mp_detections

        keypoints_0 = []
        keypoints_1 = []

        if body_part == "pose":
            detected_landmarks_0 = landmarks_0.pose_landmarks
            detected_landmarks_1 = landmarks_1.pose_landmarks

            num_of_keypoints = len(self.face_keypoints)
            key_points = self.pose_keypoints
            keypoints_history_0 = self.pose_keypoints_0_history
            keypoints_history_1 = self.pose_keypoints_1_history

        elif body_part == "face":
            detected_landmarks_0 = landmarks_0.face_landmarks
            detected_landmarks_1 = landmarks_1.face_landmarks

            num_of_keypoints = len(self.face_keypoints)
            key_points = self.face_keypoints
            keypoints_history_0 = self.face_keypoints_0_history
            keypoints_history_1 = self.face_keypoints_1_history

        elif body_part == "left_hand":
            detected_landmarks_0 = landmarks_0.left_hand_landmarks
            detected_landmarks_1 = landmarks_1.left_hand_landmarks

            num_of_keypoints = len(self.hand_keypoints)
            key_points = self.hand_keypoints
            keypoints_history_0 = self.l_hand_keypoints_0_history
            keypoints_history_1 = self.l_hand_keypoints_1_history

        elif body_part == "right_hand":
            detected_landmarks_0 = landmarks_0.right_hand_landmarks
            detected_landmarks_1 = landmarks_1.right_hand_landmarks

            num_of_keypoints = len(self.hand_keypoints)
            key_points = self.hand_keypoints
            keypoints_history_0 = self.r_hand_keypoints_0_history
            keypoints_history_1 = self.r_hand_keypoints_1_history

        else:
            print(
                "ERROR:\nbody_part must be in ['pose', 'face', 'left_hand', 'right_hand']"
            )
            return None

        if detected_landmarks_0 and detected_landmarks_1:
            for i, landmark in enumerate(detected_landmarks_0.landmark):
                #if i not in key_points: continue  
                pxl_x = landmark.x * frame_0.shape[1]
                pxl_y = landmark.y * frame_0.shape[0]
                pxl_x = int(round(pxl_x))
                pxl_y = int(round(pxl_y))
                kpts = [pxl_x, pxl_y]
                keypoints_0.append(kpts)

            for i, landmark in enumerate(detected_landmarks_1.landmark):
                #if i not in key_points: continue  
                pxl_x = landmark.x * frame_1.shape[1]
                pxl_y = landmark.y * frame_1.shape[0]
                pxl_x = int(round(pxl_x))
                pxl_y = int(round(pxl_y))
                kpts = [pxl_x, pxl_y]
                keypoints_1.append(kpts)            

            if body_part == "pose":
                self.pose_keypoints_0_history = keypoints_0
                self.pose_keypoints_1_history = keypoints_1
                self.no_detection_counter_pose = 0

            if body_part == "face":
                self.face_keypoints_0_history = keypoints_0
                self.face_keypoints_1_history = keypoints_1
                self.no_detection_counter_face = 0

            if body_part == "left_hand":
                self.l_hand_keypoints_0_history = keypoints_0
                self.l_hand_keypoints_1_history = keypoints_1
                self.no_detection_counter_l_hand = 0
                    
            if body_part == "right_hand":
                self.r_hand_keypoints_0_history = keypoints_0
                self.r_hand_keypoints_1_history = keypoints_1
                self.no_detection_counter_r_hand = 0

        else:
            # if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            if body_part == "pose":
                self.no_detection_counter_pose += 1
                
                if self.no_detection_counter_pose > self.no_detection_max:
                    keypoints_0 = [[-1, -1]] * num_of_keypoints
                    keypoints_1 = [[-1, -1]] * num_of_keypoints
                else:
                    keypoints_0 = keypoints_history_0  
                    keypoints_1 = keypoints_history_1

            if body_part == "face":
                self.no_detection_counter_face += 1
                if self.no_detection_counter_face > self.no_detection_max:
                    keypoints_0 = [[-1, -1]] * num_of_keypoints
                    keypoints_1 = [[-1, -1]] * num_of_keypoints
                else:
                    keypoints_0 = keypoints_history_0
                    keypoints_1 = keypoints_history_1

            if body_part == "left_hand":
                self.no_detection_counter_l_hand += 1
                if self.no_detection_counter_0_l_hand > self.no_detection_max:
                    keypoints_0 = [[-1, -1]] * num_of_keypoints
                    keypoints_1 = [[-1, -1]] * num_of_keypoints
                else:
                    keypoints_0 = keypoints_history_0
                    keypoints_1 = keypoints_history_1


            if body_part == "right_hand":
                self.no_detection_counter_r_hand += 1
                if self.no_detection_counter_0_r_hand > self.no_detection_max:
                    keypoints_0 = [[-1, -1]] * num_of_keypoints
                    keypoints_1 = [[-1, -1]] * num_of_keypoints
                else:
                    keypoints_0 = keypoints_history_0
                    keypoints_1 = keypoints_history_1

        return keypoints_0, keypoints_1

    def keypoints_xy(self, mp_detection, frame, camera, body_part="face"):
        keypoints = []

        if body_part == "pose":
            detected_landmarks = mp_detection.pose_landmarks
            num_of_keypoints = len(self.face_keypoints)
            key_points = self.pose_keypoints
            if camera == 0:
                keypoints_history = self.pose_keypoints_0_history
            elif camera == 1:
                keypoints_history = self.pose_keypoints_1_history

        elif body_part == "face":
            detected_landmarks = mp_detection.face_landmarks
            num_of_keypoints = len(self.face_keypoints)
            key_points = self.face_keypoints
            if camera == 0:
                keypoints_history = self.face_keypoints_0_history
            elif camera == 1:
                keypoints_history = self.face_keypoints_1_history

        elif body_part == "left_hand":
            detected_landmarks = mp_detection.left_hand_landmarks
            num_of_keypoints = len(self.hand_keypoints)
            key_points = self.hand_keypoints
            if camera == 0:
                keypoints_history = self.l_hand_keypoints_0_history
            elif camera == 1:
                keypoints_history = self.l_hand_keypoints_1_history

        elif body_part == "right_hand":
            detected_landmarks = mp_detection.right_hand_landmarks
            num_of_keypoints = len(self.hand_keypoints)
            key_points = self.hand_keypoints
            if camera == 0:
                keypoints_history = self.r_hand_keypoints_0_history
            elif camera == 1:
                keypoints_history = self.r_hand_keypoints_1_history

        else:
            print(
                "ERROR:\nbody_part must be in ['pose', 'face', 'left_hand', 'right_hand']"
            )
            return None

        if detected_landmarks:
            for i, landmark in enumerate(detected_landmarks.landmark):
                if i not in key_points:
                    continue  
                pxl_x = landmark.x * frame.shape[1]
                pxl_y = landmark.y * frame.shape[0]
                pxl_x = int(round(pxl_x))
                pxl_y = int(round(pxl_y))
                kpts = [pxl_x, pxl_y]
                keypoints.append(kpts)

            if body_part == "pose":
                if camera == 0:
                    self.pose_keypoints_0_history = keypoints
                    self.no_detection_counter_0_pose = 0
                elif camera == 1:
                    self.pose_keypoints_1_history = keypoints
                    self.no_detection_counter_1_pose = 0

            if body_part == "face":
                if camera == 0:
                    self.face_keypoints_0_history = keypoints
                    self.no_detection_counter_0_face = 0
                elif camera == 1:
                    self.face_keypoints_1_history = keypoints
                    self.no_detection_counter_1_face = 0

            if body_part == "left_hand":
                if camera == 0:
                    self.l_hand_keypoints_0_history = keypoints
                    self.no_detection_counter_0_l_hand = 0
                elif camera == 1:
                    self.l_hand_keypoints_1_history = keypoints
                    self.no_detection_counter_1_l_hand = 0
                    
            if body_part == "right_hand":
                if camera == 0:
                    self.r_hand_keypoints_0_history = keypoints
                    self.no_detection_counter_0_r_hand = 0
                elif camera == 1:
                    self.r_hand_keypoints_1_history = keypoints
                    self.no_detection_counter_1_r_hand = 0

        else:
            # if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            if body_part == "pose":
                if camera == 0:
                    self.no_detection_counter_0_pose += 1
                    if self.no_detection_counter_0_pose > self.no_detection_max:
                        keypoints = [[-1, -1]] * num_of_keypoints
                    else:
                        keypoints = keypoints_history
                elif camera == 1:
                    self.no_detection_counter_1_pose += 1
                    if self.no_detection_counter_1_pose > self.no_detection_max:
                        keypoints = [[-1, -1]] * num_of_keypoints   
                    else:
                        keypoints = keypoints_history       

            if body_part == "face":
                if camera == 0:
                    self.no_detection_counter_0_face += 1
                    if self.no_detection_counter_0_face > self.no_detection_max:
                        keypoints = [[-1, -1]] * num_of_keypoints
                    else:
                        keypoints = keypoints_history
                elif camera == 1:
                    self.no_detection_counter_1_face += 1
                    if self.no_detection_counter_1_face > self.no_detection_max:
                        keypoints = [[-1, -1]] * num_of_keypoints
                    else:
                        keypoints = keypoints_history

            if body_part == "left_hand":
                if camera == 0:
                    self.no_detection_counter_0_l_hand += 1
                    if self.no_detection_counter_0_l_hand > self.no_detection_max:
                        keypoints = [[-1, -1]] * num_of_keypoints
                    else:
                        keypoints = keypoints_history
                elif camera == 1:
                    self.no_detection_counter_1_l_hand += 1
                    if self.no_detection_counter_1_l_hand > self.no_detection_max:
                        keypoints = [[-1, -1]] * num_of_keypoints
                    else:
                        keypoints = keypoints_history

            if body_part == "right_hand":
                if camera == 0:
                    self.no_detection_counter_0_r_hand += 1
                    if self.no_detection_counter_0_r_hand > self.no_detection_max:
                        keypoints = [[-1, -1]] * num_of_keypoints
                    else:
                        keypoints = keypoints_history
                elif camera == 1:
                    self.no_detection_counter_1_r_hand += 1
                    if self.no_detection_counter_1_r_hand > self.no_detection_max:
                        keypoints = [[-1, -1]] * num_of_keypoints
                    else:
                        keypoints = keypoints_history

            

            #keypoints = [[-1, -1]] * num_of_keypoints

        return keypoints

    def plot_keypoints(self, frame, keypoints, color=(0, 0, 255), radius=3):
        if keypoints is not None:
            for point in keypoints:
                cv.circle(
                    frame, (point[0], point[1]), radius, color, -1
                )  # add keypoint detection points into figure

    def detect_keypoints(self, frame, model):
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # the BGR image to RGB.
        frame.flags.writeable = False  # pass by reference.
        results = model.process(frame)  # Detect Keypoints
        # reverse changes
        frame.flags.writeable = True
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        return frame, results

    def crop_frame(self, frame):
        """
        crop from [720, 1280] to 720x720.

        Args:
            frame (MatLike): Original frame.

        Returns:
            croped_frame (MatLike): Croped frame to 720x720

        Note: camera calibration parameters are set to this resolution.
        If you change this, make sure to also change camera intrinsic parameters
        """

        if frame.shape[1] != 720:
            croped_frame = frame[
                :,
                frame.shape[1] // 2
                - frame.shape[0] // 2 : frame.shape[1] // 2
                + frame.shape[0] // 2,
            ]

        return croped_frame


if __name__ == "__main__":

    #POSE = ["head", "upper_body", "left_hand", "right_hand", "lower_body"]
    POSE = ["upper_body", "lower_body"]
    #POSE = []
    mocap = Mocap3D(pose=POSE, hands=True, face=True)

    """
    # put camera id as command line arguements
    if len(sys.argv) == 3:
        input_stream1 = int(sys.argv[1])
        input_stream2 = int(sys.argv[2])
    else:
        exit()
    """

    # this will load the sample videos if no camera ID is given
    input_stream1 = "media\camera_0.avi"
    input_stream2 = "media\camera_1.avi"

    # get projection matrices
    P0 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)

    kpts_3d_face, kpts_3d_left_hand, kpts_3d_right_hand, kpts_3d_body = mocap.run_mp(input_stream1, input_stream2, P0, P1)



    # this will create keypoints file in current working folder
    write_keypoints_to_disk("key_points/kpts_3d_face.dat", kpts_3d_face)
    write_keypoints_to_disk("key_points/kpts_3d_left_hand.dat", kpts_3d_left_hand)
    write_keypoints_to_disk("key_points/kpts_3d_right_hand.dat", kpts_3d_right_hand)
    write_keypoints_to_disk("key_points/kpts_3d_body.dat", kpts_3d_body)
