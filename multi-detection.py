import cv2
from queue import Queue
import threading
import time
from imageai.Detection.Custom import CustomObjectDetection
import os


q = Queue(4)
execution_path = os.getcwd()


def handle_receive(cam, url):
    global q
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    fps = 3
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if time.time() - start_time > 1. / fps:
                if frame.shape[1] > 1000:
                    frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
                q.put([cam, frame])
                start_time = time.time()
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


def detection():
    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(detection_model_path=os.path.join(execution_path, "detection_model-ex-33--loss-4.97.h5"))
    detector.setJsonPath(configuration_json=os.path.join(execution_path, "detection_config.json"))
    detector.loadModel()
    while True:
        cam, image = q.get()
        detected_image, detected_data = detector.detectObjectsFromImage(input_image=image,
                                        input_type="array",
                                        output_type="array",
                                        minimum_percentage_probability=40)
        cv2.imshow(f"video : {cam}", detected_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print(image.shape)
    # do something


def main():
    urls = {0: "rtsp://root:.dkansk,,@192.168.0.105:554/stream_ch00_0"}
    detection_thread = threading.Thread(target=detection)
    detection_thread.daemon = True
    detection_thread.start()
    for cam in urls:
        recv_thread = threading.Thread(target=handle_receive, args=(cam, urls[cam]))
        recv_thread.daemon = True
        recv_thread.start()
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
