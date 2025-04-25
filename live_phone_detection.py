import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO(r"C:\Users\KIIT\OneDrive\Desktop\phone-detection\best.pt")  # Make sure 'best.pt' is in the same folder or provide full path

# Open the webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam.")
    exit()

print("Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Run prediction on the frame
    results = model.predict(source=frame, conf=0.5, stream=True)

    # Render results on the frame
    for result in results:
        annotated_frame = result.plot()
        cv2.imshow("YOLOv8 Phone Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
