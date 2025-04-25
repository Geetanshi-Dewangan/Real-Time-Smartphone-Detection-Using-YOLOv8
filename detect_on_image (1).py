import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO("best.pt")

# Load your input image (change path here)
image_path = r"C:\Users\KIIT\OneDrive\Desktop\phone-detection\greatt.jpg"  # 🔁 Put your image file here
image = cv2.imread(image_path)

# Run inference
#results = model(image, conf=0.15)
results = model(image, conf=0.15, iou=0.4)


# Draw boxes on the image
annotated_frame = results[0].plot()

# Show the result
cv2.imshow("Phone Detection", annotated_frame)

# Wait for a key press and close window
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
cv2.imwrite("output_with_detections.jpg", annotated_frame)
print("Detection saved as output_with_detections.jpg")
