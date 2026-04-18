from ultralytics import YOLO
import cv2

import glob
import os

from constant import *

class ModelInference:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = YOLO(self.model_path)
    
    def predict(self, img, conf=None):
        results = self.model(img, conf=conf)
        return results

    def get_boxes_coordinate(self, results):
        boxes_list = []
        for result in results:
            boxes_list.append(result.boxes.xyxy.tolist())
        
        return boxes_list
    
    def get_result_class(self, results):
        result_classes_list = []
        for result in results:
            result_classes_list.append(result.boxes.cls.tolist())
        
        return result_classes_list
    
    def get_orig_img(self, results):
        orig_img_list = []
        for result in results:
            orig_img_list.append(result.orig_img)
        
        return orig_img_list
    
    def get_class_names(self, results):
        for result in results:
            class_names = result.names
        
        return class_names
    
    def get_conf(self, results):
        conf_list = []
        for result in results:
            conf_list.append(result.boxes.conf.tolist())
        
        return conf_list

    def display_result(self, results):
        for result in results:
            result.show()
    
    def show_results(self, results, display=False):
        print("*" * 50)
        print("Boxes: ", self.get_boxes_coordinate(results))
        print("*" * 50)
        print("Class Names: ", self.get_class_names(results))
        print("*" * 50)
        print("Original Image Shape: ", len(self.get_orig_img(results)))
        print("*" * 50)
        print("Prediction Result: ", self.get_result_class(results))
        print("*" * 50)
        print("Prediction Confidence: ", self.get_conf(results))

        if display:
            self.display_result(results)
    
    def get_result_class_names(self, class_names, result_classes):
        result_class_names = []
        for i, result_class in enumerate(result_classes):
            class_name = []
            for cls in result_class:
                class_name.append(class_names[cls])
            result_class_names.append(class_name)
        
        return result_class_names
    
    def draw_rectangles(self, results, img_path_list , id_list = None, display=False, save=False):

        if type(img_path_list) != list:
            img = img_path_list

        boxes_coordinates = self.get_boxes_coordinate(results)
        result_class_names = self.get_result_class_names(self.get_class_names(results), self.get_result_class(results))

        img_list = []

        for i, box_coord in enumerate(boxes_coordinates):

            if id_list is not None: id = id_list[i]
            else: id = "Default"
            
            for box in box_coord:
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, str(id) + ": " + result_class_names[i][0], (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if save:
                output_dir = os.path.join(MODEL_OUTPUT_DIR, f"model_output_{id}.jpg")
                cv2.imwrite(output_dir, img)

            if display:
                cv2.imshow("Model_output", img)

            img_list.append(img)
        
        return img_list
        

        


if __name__ == "__main__":
    model_path = os.path.join(MODEL_DIR, "best.pt")
    model_inference = ModelInference(model_path)
    #img = glob.glob(os.path.join("./output", "*"))
    img = ['./output/0.png','./output/1.png','./output/2.png','./output/3.png','./output/4.png','./output/5.png']
    results = model_inference.predict(img, conf=0.5)

    model_inference.show_results(results, isDisplay=False)
    model_inference.draw_rectangles(results, img, display=True, save=True)
        
