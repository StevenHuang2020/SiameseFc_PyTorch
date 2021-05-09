import cv2
import numpy as np
from siamfc import TrackerSiamFC

def rectangleImg(img,startPt,stopPt,color=(0,0,255),thickness=2):
    return cv2.rectangle(img, startPt, stopPt, color=color, thickness=thickness) 

def cameraTracking():
    net_path = 'siamfc_alexnet_e554.pth'
    tracker = TrackerSiamFC(net_path=net_path)

    cap = cv2.VideoCapture(0) #set tracking target boundingbox, press enter
    ret, first_frame = cap.read()
    cv2.imshow('first_frame',first_frame)
    bbi = cv2.selectROI('first_frame',first_frame)
    cv2.destroyAllWindows() 
    print('The object you select:', bbi)
    
    tracker.init(first_frame, bbi)
    
    while(True):
        ret, frame = cap.read()
               
        box = tracker.update(frame)
        #print('box=',box)
        
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[0] + box[2]), int(box[1]+box[3]))
        show = rectangleImg(frame, pt1, pt2)

        # Display the resulting frame
        cv2.imshow('frame',show)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cameraTracking()
