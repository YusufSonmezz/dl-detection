import cv2
import numpy as np

from constant import *

class VideoManipulator:

    def __init__(self, video_path):
        self.video_path = video_path

        self.tracking_objects = {}
        self.object_id = 0

        self.object_details = []
        self.saved_ids = set()  # Set to keep track of saved IDs

    def read_video(self):
        cap = cv2.VideoCapture(self.video_path)

        width, height = self.get_size_of_video()
        # Define artificial line
        artificial_line = (0, int(height / 2)), (int(width), int(height / 2))

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            frame = frame[:, 650:-350]

            ########## Silinecek ##########
            frame = frame[:, 650:-350]
            frame = cv2.resize(frame, (640, 480))
            artificial_line = (0, int(480 / 2)), (640, int(480 / 2))
            ########## Silinecek ##########

            extracted_element_dict, coord_dict = self.extract_metal_from_frame(frame, artificial_line)

            for key, value in extracted_element_dict.items():
                cv2.imshow(f'Extracted Element {key}', value)
                print(f'Extracted Element ID:{key}, Coordinates: {coord_dict[key]}')
                
                output_dir = os.path.join(OUTPUT_IMAGES_DIR, f'{key}.png')
                cv2.imwrite(output_dir, value)

                x, y, w, h = coord_dict[key]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {key}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            
            cv2.line(frame, artificial_line[0], artificial_line[1], (255, 0, 0), 2)
            cv2.imshow('Frame', frame)

            if coord_dict:
                cv2.waitKey(0)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    
    def get_size_of_video(self):
        cap = cv2.VideoCapture(self.video_path)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return width, height
    
    def is_contour_crossed_line(self, contour, line):
        # Calculate bounding rectangle for the contour
        x, y, w, h = cv2.boundingRect(contour)
        rect_points = np.array([(x, y), (x + w, y), (x, y + h), (x + w, y + h)])

        # Line equation coefficients (Ax + By + C = 0)
        (x1, y1), (x2, y2) = line
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2

        # Check each corner of the rectangle to see if it's on different sides of the line
        signs = np.sign(A * rect_points[:, 0] + B * rect_points[:, 1] + C)
        if np.any(np.diff(signs)):
            return True

        return False
    
    def get_centroid(self, contour):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        return cx, cy
    
    def process_frame(self, frame, display=False):
        """# Convert the frame to grayscale to detect edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply median blur to the frame for noise reduction
        gray_blured = cv2.medianBlur(gray, 5)
        # Detect edges in the image
        edges = cv2.adaptiveThreshold(gray_blured, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

        if display: cv2.imshow('Edges', edges)

        # Detect contours
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)"""

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        if display: cv2.imshow('Edges', edges)
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        return contours

    def extract_metal_from_frame(self, frame, artificial_line):
        processing_frame = frame.copy()
        contours = self.process_frame(processing_frame)

        unique_extracted_sheets = {}
        sheets_coordinates = {}

        current_objects = {}
        centeroid_coordinates = []

        for contour in contours:
            if cv2.contourArea(contour) < 3000:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            if x == 0 and y == 0:
                continue

            if self.is_contour_crossed_line(contour, artificial_line):
                cx, cy = self.get_centroid(contour)
                if [cx, cy] in centeroid_coordinates:
                    continue
                else: centeroid_coordinates.append([cx, cy])
                current_obj_id = None

                found = False
                for obj_id, (prev_cx, prev_cy) in self.tracking_objects.items():
                    if np.linalg.norm((cx - prev_cx, cy - prev_cy)) < 50:  # Threshold distance
                        current_objects[obj_id] = (cx, cy)
                        current_obj_id = obj_id
                        found = True
                        break

                if not found:
                    current_objects[self.object_id] = (cx, cy)
                    current_obj_id = self.object_id
                    self.object_id += 1

                    # Only save the ID if it hasn't been saved in this frame
                    if current_obj_id not in self.saved_ids:
                        cropped_frame = processing_frame[y:y + h, x:x + w]
                        unique_extracted_sheets[current_obj_id] = cropped_frame
                        sheets_coordinates[current_obj_id] = (x, y, w, h)
                        self.saved_ids.add(current_obj_id)

                if current_obj_id is not None:
                    self.object_details.append((x, y, w, h, current_obj_id))

        self.tracking_objects.update(current_objects)

        return unique_extracted_sheets, sheets_coordinates




if __name__ == '__main__':
    
    video_path = os.path.join(VIDEO_DIR, 'IMG_0729.MOV')
    video_manipulator = VideoManipulator(video_path)

    video_manipulator.read_video()
    print("Size of video: ", video_manipulator.get_size_of_video())
    #video_manipulator.extract_metal_sheets()