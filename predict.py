import cv2
import numpy as np
import os

# Paths to the model files
cfg_path = os.path.join('model', 'yolov3.cfg')
weights_path = os.path.join('model', 'yolov3.weights')
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Define the classes your model can detect
classes = ["door", "window", "sofa"]

def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image at {image_path}")
        return {}
    height, width = img.shape[:2]

    print(f"Image shape: {img.shape}")

    # Create a blob and pass it through the model
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists to hold detected class names, confidences, and bounding boxes
    class_ids = []
    confidences = []
    boxes = []

    confidence_threshold = 0.3  # Adjust this value
    nms_threshold = 0.4  # Non-max suppression threshold

    # Loop through the outputs and extract information
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                print(f"Class ID: {class_id}, Confidence: {confidence}")
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                print(f"Bounding Box: x={x}, y={y}, w={w}, h={h}")
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    
    if len(indices) == 0:
        print("No detections after non-max suppression")

    # Initialize a dictionary to hold the count and locations of each detected class
    detection_results = {class_name: {'count': 0, 'locations': []} for class_name in classes}

    # Loop through the detections after applying NMS
    for i in indices:
        i = i[0]
        box = boxes[i]
        x, y, w, h = box
        class_name = classes[class_ids[i]]
        detection_results[class_name]['count'] += 1
        detection_results[class_name]['locations'].append({'x': x, 'y': y, 'width': w, 'height': h})

    print("Detection results:", detection_results)
    return detection_results
