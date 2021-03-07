import cv2
import time
import numpy as np
import sys
from norfair import Detection, Tracker, Video, draw_tracked_objects
from shapely.geometry import Polygon

class CarCounter:
    def __init__(self):
        self.outboundArray = []
        self.inboundArray = []
        self.outboundPolygon = Polygon()
        self.inboundPolygon = Polygon()
        self.outDrawn = False
        self.inDrawn = False
        self.inCars = []
        self.outCars = []
        
    def euclidean_distance(self, detection, tracked_object):
        return np.linalg.norm(detection.points - tracked_object.estimate)

    def convertToDetection(self, detection_yolo):
        x, y, w, h = detection_yolo["detection"]  
        return Detection(
            np.array(((x+w/2, y+h/2))),
            data={"label": detection_yolo["label"], "score": detection_yolo["score"], "realbox": [x,y,w,h]},
        )

    def mouseClick(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if(len(self.outboundArray) < 4):
                self.outboundArray.append([x,y])
                if len(self.outboundArray) == 4:
                    self.outboundPolygon = Polygon([
                        (self.outboundArray[0][0],self.outboundArray[0][1]),
                        (self.outboundArray[1][0],self.outboundArray[1][1]),
                        (self.outboundArray[2][0],self.outboundArray[2][1]),
                        (self.outboundArray[3][0],self.outboundArray[3][1])])
                    self.outDrawn = True

            else:
                return
        if event == cv2.EVENT_RBUTTONDOWN:
            if(len(self.inboundArray) < 4):
                self.inboundArray.append([x,y])
                if len(self.inboundArray) == 4:
                    self.inboundPolygon = Polygon([
                        (self.inboundArray[0][0],self.inboundArray[0][1]),
                        (self.inboundArray[1][0],self.inboundArray[1][1]),
                        (self.inboundArray[2][0],self.inboundArray[2][1]),
                        (self.inboundArray[3][0],self.inboundArray[3][1])])
                    self.inDrawn = True
            else:
                return
       

    def run(self):
        CONFIDENCE_THRESHOLD = 0.1
        NMS_THRESHOLD = 0.1

        class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]

        vc = cv2.VideoCapture(0)
        #vc.set(cv2.CAP_PROP_FPS,60)

        net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(size=(416, 416), scale=1/255)

        tracker = Tracker( 
            distance_function=self.euclidean_distance,
            distance_threshold=10
        )

        cv2.namedWindow("Frame")
        cv2.setMouseCallback("Frame", self.mouseClick)

        while cv2.waitKey(1) < 1:
            (grabbed, frame) = vc.read()
            if not grabbed:
                exit()

            start = time.time()
            classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
            end = time.time()

            carScores = []

            for (classid, score, box) in zip(classes, scores, boxes):
                if(classid == 2):
                    detection = {
                        "detection": (box[0],box[1],box[2],box[3]),
                        "label": class_names[int(classid)],
                        "score": score[0]
                    }
                    carScores.append(self.convertToDetection(detection))

            carDetections = tracker.update(carScores)

            start_drawing = time.time()

            if (self.outDrawn):
                cv2.polylines(frame, [np.array(self.outboundArray, np.int32)], True, (0,0,255), 2)

            if (self.inDrawn):
                cv2.polylines(frame, [np.array(self.inboundArray, np.int32)], True, (0,255,0), 2)

            for detection in carDetections:
                box = detection.last_detection.data['realbox']
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[0]+box[2])
                y2 = int(box[1]+box[3])
                carNumber = detection.id
                #detection.estimate[0][1] + 30

                label = '{}:{} score: {:.2f}'.format(detection.last_detection.data['label'], carNumber , detection.last_detection.data['score']*100)

                startPoint = (x1,y1)
                endPoint = (x2,y2)

                car = frame[y1:y2, x1:x2]
                avg_color_per_row = np.average(car, axis=0)
                avg_color = np.average(avg_color_per_row, axis=0)
                carcolor = (avg_color[0], avg_color[1], avg_color[2])
            
                rect = Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])

                if (rect.intersects(self.inboundPolygon)):
                    if not carNumber in self.inCars:
                        self.inCars.append(carNumber)
                    cv2.rectangle(frame, startPoint, endPoint, (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                elif (rect.intersects(self.outboundPolygon)):
                    if not carNumber in self.outCars:
                        self.outCars.append(carNumber)
                    cv2.rectangle(frame, startPoint, endPoint, (0, 0, 255), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame, startPoint, endPoint, carcolor, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, carcolor, 2)
            end_drawing = time.time()

            outLabel = "OUT: %i" % len(self.outCars)
            inLabel = "IN: %i" % len(self.inCars)
            fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000)
            cv2.putText(frame, outLabel, (1700, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 0, 255),3)
            cv2.putText(frame, inLabel, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 0), 3)
            cv2.putText(frame, fps_label, (0, 1000), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.imshow("Frame", frame)
        
if __name__ == '__main__':
    counter = CarCounter()
    counter.run()
    

