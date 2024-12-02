import cv2
import numpy as np
from pyzbar.pyzbar import decode
import time

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width of video frame
cap.set(4, 480)  # Set height of video frame

last_print_time = time.time()  # Record the initial time

while True:
    success, img = cap.read()
    
    for barcode in decode(img):
        myData = barcode.data.decode('utf-8')
        current_time = time.time()
        
        # Print myData only if 2 seconds have passed since the last print
        if current_time - last_print_time >= 2:
            print(myData)
            last_print_time = current_time  # Update the last print time

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

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
