import threading
from image_extraction import VideoManipulator
from constant import *
from model_inference import ModelInference
import cv2
import os

class Processor:
    def __init__(self, video_path, model_path):
        self.video_path = video_path
        self.model_path = model_path

        self.video_manipulator = VideoManipulator(self.video_path)
        self.model_inference = ModelInference(self.model_path)

        self.latest_frame = None
        self.latest_results = None
        self.latest_extracted_frame = None
        self.latest_id = None

        self.lock = threading.Lock()
    
    def read_video_as_frame(self):
        cap = cv2.VideoCapture(self.video_path)
        width, height = self.video_manipulator.get_size_of_video()
        artificial_line = (0, int(height / 2)), (int(width), int(height / 2))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = frame[:, 650:-350]
            w, h = 1080, 1920
            frame = cv2.resize(frame, (h, w))
            artificial_line = (0, int(w / 2)), (h, int(w / 2))

            with self.lock:
                self.latest_frame = frame.copy()

            extracted_element_dict, coord_dict = self.video_manipulator.extract_metal_from_frame(frame, artificial_line)

            for key, value in extracted_element_dict.items():
                print(f'Extracted Element ID:{key}, Coordinates: {coord_dict[key]}')
                
                output_dir = os.path.join(OUTPUT_IMAGES_DIR, f'{key}.png')
                cv2.imwrite(output_dir, value)

                x, y, w, h = coord_dict[key]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {key}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                with self.lock:
                    self.latest_results = self.model_inference.predict(value, conf=0.5)
                    self.latest_extracted_frame = value.copy()
                    self.latest_id = key

            cv2.line(frame, artificial_line[0], artificial_line[1], (255, 0, 0), 2)
            cv2.imshow('Video Stream', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def display_results(self):
        while True:
            with self.lock:
                if self.latest_extracted_frame is not None and self.latest_results is not None:
                    frame_copy = self.latest_extracted_frame.copy()
                    self.model_inference.draw_rectangles(self.latest_results, frame_copy, id_list=[self.latest_id], display=True, save=True)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    video_path = os.path.join(VIDEO_DIR, "IMG_0729.mov")
    model_path = os.path.join(MODEL_DIR, "best.pt")
    processor = Processor(video_path, model_path)

    video_thread = threading.Thread(target=processor.read_video_as_frame)
    display_thread = threading.Thread(target=processor.display_results)

    video_thread.start()
    display_thread.start()

    video_thread.join()
    display_thread.join()
