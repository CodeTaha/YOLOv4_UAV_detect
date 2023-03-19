import cv2
import numpy as np

cap = cv2.VideoCapture("images/VID-20221027-WA0002(0).mp4")
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    
    ret, frame = cap.read()

    model = cv2.dnn.readNetFromDarknet("model/spot_yolov4.cfg", "model/spot_yolov4_last.weights")
    layers = model.getLayerNames()
    unconnect = model.getUnconnectedOutLayers()
    unconnect = unconnect-1
    
    output_layers = []
    for i in unconnect:
        output_layers.append(layers[int(i)])
    
    classFile = 'spot.names'# dikkat et
    classNames = []
    
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
    
    
    frame = cv2.resize(frame, (1600,900))
    
    img_width = frame.shape[1]
    img_height = frame.shape[0]
    
    img_blob = cv2.dnn.blobFromImage(frame,1/255,(416,416),swapRB=True)
    
    model.setInput(img_blob)
    detection_layers = model.forward(output_layers)
    
    ids_list = []
    boxes_list = []
    confidences_list = []
    
    for detection_layer in detection_layers:
        for object_detection in detection_layer:
            scores = object_detection[5:]
            predicted_id = np.argmax(scores)
            confidence = scores[predicted_id]
            
            if confidence > 0.30:
                label = classNames[predicted_id]
                bounding_box = object_detection[0:4] * np.array([img_width,img_height,img_width,img_height])
                (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")
                start_x = int(box_center_x - (box_width/2))
                start_y = int(box_center_y - (box_height/2))
                
                ids_list.append(predicted_id)
                confidences_list.append(float(confidence))
                boxes_list.append([start_x, start_y, int(box_width), int(box_height)])
              
    max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)
    
    for max_id in max_ids:
        max_class_id = max_id
        box = boxes_list[max_class_id]
        
        start_x = box[0]
        start_y = box[1]
        box_width = box[2]
        box_height = box[3]
        
        predicted_id = ids_list[max_class_id]
        confidence = confidences_list[max_class_id]
        
        end_x = start_x + box_width
        end_y = start_y + box_height
        cenx = start_x + ((end_x-start_x)/2)
        ceny = start_y + ((end_y-start_y)/2)
        alan = (end_x-start_x)*(end_y-start_y)
        frsize = img_height*img_width
        hedefBoyutu = (alan*100)/frsize
        
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0,0,255), 2)
        
        cv2.line(frame, (788,450), (812,450), (0,255,0), thickness = 2)
        cv2.line(frame, (800,438), (800,462), (0,255,0), thickness = 2)
        cv2.rectangle(frame, (400,90), (1200,810), (255,0,0), thickness = 3)
            
        cv2.putText(frame, "TAC/Guven: {:.2f}%".format(confidence*100), (5,20), font, 0.8, (0,0,0), cv2.LINE_4)
        cv2.putText(frame, "FPS: {}".format("47"), (5, 50), font, 0.8, (0,0,0), cv2.LINE_4)
        cv2.putText(frame, "Kilitlenme Sartlari:<<HEDEF BULUNDU>>", (5, 110), font, 0.6, (0,0,0), cv2.LINE_4)
        cv2.putText(frame, "Hedef Konumu: x = {} y = {}".format(cenx, ceny), (5, 140), font, 0.6, (0,0,0), cv2.LINE_4)
        cv2.putText(frame, "Hedef Boyutu: %{}".format(round(hedefBoyutu)), (5, 170), font, 0.7, (0,0,0), cv2.LINE_4)
        cv2.putText(frame, "Takip suresi: none", (5, 200), font, 0.7, (0,0,0), cv2.LINE_4)
        cv2.putText(frame, "BASARILI KILITLENME: 0", (5, 260), font, 0.8, (0,0,0), cv2.LINE_4)
        cv2.circle(frame, (box_center_x, box_center_y), 3, (0,255,0), 1)
        cv2.line(frame, (800,450), (box_center_x, box_center_y), (0,0,0), thickness = 2)
        
        if cv2.waitKey(1) & ord("q") == 27:
            break
        
        cv2.imshow("frame", frame)

cap.release()
cv2.destroyAllWindows()