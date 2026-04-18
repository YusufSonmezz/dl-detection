from ultralytics import YOLO


class Yolov8:

    def __init__(self):
        self.model = YOLO('yolov8s.pt')

    def return_model(self):
        return self.model