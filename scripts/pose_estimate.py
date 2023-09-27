import mediapipe as mp
import cv2
from cvfpscalc import CvFpsCalc


class PoseEstimate():
    def __init__(self, web=True) -> None:
        self.web = web

        self.fps = CvFpsCalc()
        self.init_camera()

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, 
                                                  min_tracking_confidence=0.5,
                                                  model_complexity=1)

    def init_camera(self):
        if self.web:
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture("videoplayback.mp4")


    def estimate(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      # Recolor feed
            results = self.holistic.process(image)              # Make detections
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)      # Recolor feed
            
            # Face
            self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION,
                                           self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=1),
                                           self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))
            # Right hand
            self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                           self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                           self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))
            # Left hand
            self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                           self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                           self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))
            # Body
            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                           self.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
                                           self.mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2))
            
            image = cv2.putText(image, str(self.fps.get()), org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                   fontScale=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            
            cv2.imshow("Holistic Model Detections", image)      

            if cv2.waitKey(10) & 0xFF == ord("q"):
                self.destroy_all()
                return "destroy"
            

            return results

    def destroy_all(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__name__":
    estimator = PoseEstimate()
    while True:
        if estimator.estimate() == "destroy":
            break