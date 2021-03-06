import cv2
import time
import numpy as np
import sys
sys.path.append("..")
from sort.sort import *

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Rectangle:
    def __init__(self,x1,y1,x2,y2):
        self.top = y1
        self.bottom = y2
        self.left = x1
        self.right = x2


class Line:
    def __init__(self, startPoint: Point, endPoint: Point):
        self.startPoint = startPoint
        self.endPoint = endPoint


    def calculateY(self, x):
        lineX = [self.startPoint.x, self.endPoint.x]
        lineY = [self.startPoint.y, self.endPoint.y]
        return np.interp(x, lineX, lineY)
    

class CarCounter:
    def __init__(self):
        #Line to calculate inbound cars
        self.inLineX1 = 325
        self.inLineY1 = 500
        self.inLineX2 = 650
        self.inLineY2 = 525

        #Line to calculate outbound cars
        self.outLineX1 = 650
        self.outLineY1 = 525
        self.outLineX2 = 875
        self.outLineY2 = 550

        self.inStartPoint = Point(self.inLineX1,self.inLineY1)
        self.inEndPoint = Point(self.inLineX2,self.inLineY2)

        self.outStartPoint = Point(self.outLineX1,self.outLineY1)
        self.outEndPoint = Point(self.outLineX2,self.outLineY2)

        self.outCars = []
        self.inCars = []

        
    def isRectangleOverlap(self, pointA: Point, pointB: Point, r: Rectangle):
        
        line = Line(pointA, pointB)

        if (r.left > line.endPoint.x or r.right < line.startPoint.x ):
            return False

        if (r.top < line.startPoint.y or r.bottom > line.endPoint.y ):
            return False

        yAtRectLeft = line.calculateY(r.left)
        yAtRectRight = line.calculateY(r.right)

        if (r.bottom > yAtRectLeft and r.bottom > yAtRectRight):
            return False

        if (r.top < yAtRectLeft and r.top < yAtRectRight):
            return False

        return True

    def run(self):
        CONFIDENCE_THRESHOLD = 0.1
        NMS_THRESHOLD = 0.1

        #3600 is used to calculate cars. This is the ammount of frames the sorting algorithm uses
        carTracker = Sort(3600,5,0.4)
        #personTracker = Sort(3600,5,0.4)

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

        net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(size=(416, 416), scale=1/255)

        while cv2.waitKey(1) < 1:
            (grabbed, frame) = vc.read()
            if not grabbed:
                exit()

            start = time.time()
            classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
            end = time.time()

            carScores = []
            personScores = []

            for (classid, score, box) in zip(classes, scores, boxes):
                boxScore = [box[0],box[1],box[0]+box[2],box[1]+box[3],score[0]]
                # We are only interested in cars, but it can be used for everything offcourse
                if (classid == 2):
                    carScores.append(boxScore)
                # elif (classid == 0):
                #     personScores.append(boxScore)

            if len(carScores) > 0:
                carBoxes = carTracker.update(np.array(carScores))
            else:
                carBoxes = carTracker.update(np.empty((0, 5))) 

            # if len(personScores) > 0:
            #     personBoxes = personTracker.update(np.array(personScores))
            # else:
            #     personBoxes = personTracker.update(np.empty((0, 5))) 
            
            start_drawing = time.time()

            for box in carBoxes:
                carNumber = box[4]
                label = "car number: %f" % (carNumber)
                x1 = int(box[2])
                y1 = int(box[3])
                x2 = int(box[0])
                y2 = int(box[1])

                startPoint = (x1,y1)
                endPoint = (x2,y2)

                rect = Rectangle(x1,y1,x2,y2)

                if (self.isRectangleOverlap(self.inStartPoint, self.inEndPoint, rect)):
                    if not carNumber in self.inCars:
                        self.inCars.append(carNumber)
                    cv2.rectangle(frame, startPoint, endPoint, (0, 0, 255), 2)
                    cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                elif (self.isRectangleOverlap(self.outStartPoint, self.outEndPoint, rect)):
                    if not carNumber in self.outCars:
                        self.outCars.append(carNumber)
                    cv2.rectangle(frame, startPoint, endPoint, (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, startPoint, endPoint, (255, 0, 0), 2)
                    cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # for box in personBoxes:
            #     x1 = int(box[2])
            #     y1 = int(box[3])
            #     x2 = int(box[0])
            #     y2 = int(box[1])

            #     startPoint = (x1,y1)
            #     endPoint = (x2,y2)

            #     label = "person number: %f" % (box[4])
            #     cv2.rectangle(frame, startPoint, endPoint, (255, 0, 255), 2)
            #     cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            end_drawing = time.time()

            cv2.line(frame, (self.inLineX1, self.inLineY1), (self.inLineX2, self.inLineY2), (0,0,255), 3)
            cv2.line(frame, (self.outLineX1, self.outLineY1), (self.outLineX2, self.outLineY2), (0,255,0), 3)
            outLabel = "OUT: %i" % len(self.outCars)
            inLabel = "IN: %i" % len(self.inCars)
            fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000)
            cv2.putText(frame, outLabel, (1700, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 0, 255),3)
            cv2.putText(frame, inLabel, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 0), 3)
            cv2.putText(frame, fps_label, (0, 1000), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.imshow("detections", frame)
        
if __name__ == '__main__':
    counter = CarCounter()
    counter.run()
    

