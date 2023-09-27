import cv2

def list_cameras():
    index = 0
    arr = []
    while index < 5:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

if __name__ == "__main__":
    camera_list = list_cameras()
    print(camera_list)