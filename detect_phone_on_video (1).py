import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO(r'C:\Users\KIIT\OneDrive\Desktop\phone-detection\best.pt')  # Path to your trained YOLOv8 model

# Path to the input video file
video_path = r"C:\Users\KIIT\OneDrive\Desktop\phone-detection\manTalking2.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter to save output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('ManTalkingOutput222.mp4', fourcc, fps, (width, height))

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 prediction
    results = model.predict(source=frame, conf=0.3, verbose=False)

    # Plot results on frame
    annotated_frame = results[0].plot()

    # Show frame
    cv2.imshow('Phone Detection - Press Q to Quit', annotated_frame)

    # Save frame
    out.write(annotated_frame)

    # Quit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
