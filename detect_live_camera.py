
import cv2
import math
import sys
from ultralytics import YOLO

def main():
    print("Initializing Real-Time Helmet Detection System...")

    # Load the YOLO model. 
    # We will use your trained model from the desktop project directory.
    model_path = r'c:\Users\USER\OneDrive\Desktop\helmet detection project\helmet.pt'
    
    try:
        model = YOLO(model_path)
        print(f"Successfully loaded model: {model_path}")
    except Exception as e:
        print(f"Error loading {model_path}. \nDownloading the default yolov8n.pt (You should replace this with a helmet-specific model like best.pt soon!)")
        model = YOLO('yolov8n.pt') # Fallback to standard YOLOv8 nano model if best.pt is missing

    # Class names of the model. 
    # Getting dynamic names directly from your trained model:
    classNames = model.names

    # Define colors for each class (Green for Helmet, Red for No Helmet)
    colors = {
        "With Helmet": (0, 255, 0),       # Green box
        "Without Helmet": (0, 0, 255)     # Red box
    }

    # Open the webcam (0 is usually the default laptop camera)
    # Using cv2.CAP_DSHOW to prevent MSMF camera grab errors on Windows
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Set Width and Height for WebCam (Optional)
    cap.set(3, 1280) # Width
    cap.set(4, 720)  # Height

    if not cap.isOpened():
        print("Error: Could not open the webcam. Please ensure your camera is connected.")
        sys.exit()

    print("Camera initialized successfully. Press 'q' to quit the live demo.")

    while True:
        success, img = cap.read()
        
        if not success:
            print("Failed to grab frame from camera. Exiting...")
            break

        # Run inference on the current frame
        results = model(img, stream=True, verbose=False)

        # Process the results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get the Bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 

                # Get Confidence score (Probability that the AI is correct)
                conf = math.ceil((box.conf[0] * 100)) / 100

                # Get Class index (0 for Helmet, 1 for No Helmet, etc.)
                cls = int(box.cls[0])
                
                # Protect against out of bounds or missing keys
                currentClass = classNames.get(cls, "Unknown")

                # Check if confidence is above a reasonable threshold (e.g., 50%)
                if conf > 0.5:
                    # Set color based on the class (or Cyan if we don't recognize the class index)
                    color = colors.get(currentClass, (255, 255, 0)) 
                    
                    # 1. Draw the bounding box around the detected person/head
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

                    # 2. Add standard text displaying class name and confidence
                    text = f"{currentClass} {conf}"
                    
                    # 3. Draw a filled rectangle behind text for better readability on video
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                    cv2.rectangle(img, (x1, y1 - 35), (x1 + text_size[0], y1), color, cv2.FILLED)
                    cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Show the processed frame on the screen
        cv2.imshow("Real-Time Helmet Detection Demo", img)

        # Wait for key press. If 'q' is pressed, exit the infinite loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Application terminated by user.")
            break

    # Important: Always release system resources when finished
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
