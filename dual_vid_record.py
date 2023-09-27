import cv2 as cv


class DualVidRecorder():
    def __init__(self, fps=30, resolution=(720,1280)) -> None:
        
        self.fps = fps
        self.resolution = resolution

        self.video_writer0 = cv.VideoWriter('media/camera_0.avi', 
                             cv.VideoWriter_fourcc(*'MJPG'),
                             self.fps, (self.resolution[1], self.resolution[0]))

        self.video_writer1 = cv.VideoWriter('media/camera_1.avi', 
                             cv.VideoWriter_fourcc(*'MJPG'),
                             self.fps, (self.resolution[1], self.resolution[0]))
        
        self.scale_percent = 60 # percent of original size
        self.width_resizes = int(self.resolution[1] * self.scale_percent / 100)
        self.height_resized = int(self.resolution[0] * self.scale_percent / 100)
        
        # input video stream
        self.cap0 = cv.VideoCapture(0)
        self.cap1 = cv.VideoCapture(1)

        self.caps = [self.cap0, self.cap1]

        # set camera resolution if using webcam to 1280x720.
        for cap in self.caps:
            cap.set(3, self.resolution[1])
            cap.set(4, self.resolution[0])

    def record(self):
        while True:
            # read frames from stream
            ret0, frame0 = self.cap0.read()
            ret1, frame1 = self.cap1.read()

            if not ret0 or not ret1:
                break

            self.video_writer0.write(frame0)
            self.video_writer1.write(frame1)
            

            resized0 = cv.resize(frame0, (self.width_resizes, self.height_resized), interpolation = cv.INTER_AREA)
            resized1 = cv.resize(frame1, (self.width_resizes, self.height_resized), interpolation = cv.INTER_AREA)

            cv.imshow("cam0", resized0)
            cv.imshow("cam1", resized1)

            k = cv.waitKey(1)
            if k & 0xFF == 27: break  # 27 is ESC key



        self.release()

    def release(self):
        for cap in self.caps:
            cap.release()
        self.video_writer0.release()
        self.video_writer1.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    recorder = DualVidRecorder()
    recorder.record()