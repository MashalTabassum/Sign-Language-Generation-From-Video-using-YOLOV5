import cv2
import numpy as np

# Load the pre-trained model
model = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

# Load the class names
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Load the image
image = cv2.imread('images/image.jpg')
height, width, _ = image.shape

# Prepare the image for the model
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
model.setInput(blob)

# Get the output layer names
layer_names = model.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in model.getUnconnectedOutLayers()]

# Run the model
outs = model.forward(output_layers)

# Process the outputs
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  # Confidence threshold
            # Get the bounding box coordinates
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Draw the bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{classes[class_id]}: {confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show the output
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()