import cv2
import numpy as np
from pyzbar.pyzbar import decode
import pytesseract
import time
import pandas as pd
from datetime import datetime
import os

# Create folder to save snapshots if it doesn't exist
snapshot_folder = "snapshots"
if not os.path.exists(snapshot_folder):
    os.makedirs(snapshot_folder)

# Start video capture
cap = cv2.VideoCapture(0)

address = "https://192.168.31.145:8080/video"
cap.open(address)
cap.set(3, 640)  # Set width of video frame
cap.set(4, 480)  # Set height of video frame

last_print_time = time.time()  # Record the initial time

# Create an empty list to store data for Excel
data_list = []

while True:
    success, img = cap.read()
    img = cv2.resize(img, (720,720))
    
    if not success:
        print("Failed to capture image")
        break  # Exit the loop if image capture fails

    for barcode in decode(img):
        myData = barcode.data.decode('utf-8')
        current_time = time.time()

        # Print myData only if 2 seconds have passed since the last print
        if current_time - last_print_time >= 2:
            print("Barcode Data:", myData)
            pts = np.array([barcode.polygon], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, (255, 0, 255), 5)
            pts2 = barcode.rect
            cv2.putText(img, myData, (pts2[0], pts2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
            last_print_time = current_time  # Update the last print time

            # Preprocess the frame for better OCR results
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)  # Apply thresholding

            # Extract text from the preprocessed frame
            extracted_text = pytesseract.image_to_string(thresh)
            print("Extracted Text from Frame:")
            print(extracted_text)

            # Get the bounding boxes for each character detected by Tesseract
            h, w, _ = img.shape
            boxes = pytesseract.image_to_boxes(thresh)

            # Draw boxes around the text identified by Tesseract
            for b in boxes.splitlines():
                b = b.split()
                x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
                cv2.rectangle(img, (x, h), (w, y), (0, 255, 0), 2)

            # Get the current timestamp and replace ':' with '_'
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # Add the barcode data, extracted text, and timestamp to the list
            data_list.append([myData, extracted_text, timestamp])

            # Save a snapshot of the frame when barcode data is detected
            snapshot_filename = f"{snapshot_folder}/snapshot_{timestamp}.jpg"
            cv2.imwrite(snapshot_filename, img)
            print(f"Snapshot saved as: {snapshot_filename}")

        # Draw a polygon around the barcode and display the data on the image
        pts = np.array([barcode.polygon], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (255, 0, 255), 5)
        pts2 = barcode.rect
        cv2.putText(img, myData, (pts2[0], pts2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    # Display the result in a window
    cv2.imshow('Result', img)

    # Wait for a key press; break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Create a DataFrame from the data list
df = pd.DataFrame(data_list, columns=["Barcode Data", "Extracted Text", "Timestamp"])

# Save the DataFrame to an Excel file
df.to_excel("extracted_data.xlsx", index=False)

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
