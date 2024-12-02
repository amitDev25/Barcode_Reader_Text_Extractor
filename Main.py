import cv2
import numpy as np

cap = cv2.VideoCapture(0)

DIGITS_LOOKUP = {
    (1, 1, 1, 1, 1, 1, 0): 0,
    (1, 1, 0, 0, 0, 0, 0): 1,
    (1, 0, 1, 1, 0, 1, 1): 2,
    (1, 1, 1, 0, 0, 1, 1): 3,
    (1, 1, 0, 0, 1, 0, 1): 4,
    (0, 1, 1, 0, 1, 1, 1): 5,
    (0, 1, 1, 1, 1, 1, 1): 6,
    (1, 1, 0, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 0, 1, 1, 1): 9,
}

H_W_Ratio = 1.9

def preprocess(img, threshold=12):
    # Adaptive thresholding to binarize the image
    dst = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 127, threshold)
    
    # Morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel)
    
    return dst

def find_digits_positions(img):
    # Sum the pixel values along the horizontal axis (columns)
    img_array = np.sum(img, axis=0)
    threshold = np.max(img_array) // 2
    digits_positions = []
    
    # Find the continuous segments that are likely digits
    start = None
    for i, val in enumerate(img_array):
        if val > threshold:
            if start is None:
                start = i
        elif start is not None:
            digits_positions.append((start, i))
            start = None
    return digits_positions

def recognize_digits(digits_positions, img):
    digits = []
    for (x0, x1) in digits_positions:
        roi = img[:, x0:x1]  # Extract region of interest (ROI)
        
        # Resize ROI to standard size for easier digit recognition
        resized_roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Binarize the resized ROI
        _, binarized_roi = cv2.threshold(resized_roi, 128, 255, cv2.THRESH_BINARY_INV)
        
        # Extract the digit using predefined 7-segment digit patterns
        on = [0] * 7
        segments = [
            (0, 2, 0, 14),  # Top segment
            (2, 6, 0, 2),   # Top-left segment
            (2, 6, 2, 14),  # Bottom-left segment
            (6, 8, 0, 14),  # Bottom segment
            (2, 6, 6, 8),   # Top-right segment
            (6, 8, 2, 14),  # Bottom-right segment
            (4, 6, 0, 14),  # Middle segment
        ]
        
        for i, (x1, x2, y1, y2) in enumerate(segments):
            segment = binarized_roi[y1:y2, x1:x2]
            if cv2.countNonZero(segment) > 0.5 * segment.size:
                on[i] = 1
        
        # Match the recognized pattern with the predefined DIGITS_LOOKUP
        if tuple(on) in DIGITS_LOOKUP:
            digit = DIGITS_LOOKUP[tuple(on)]
        else:
            digit = '*'
        
        digits.append(digit)
    
    return digits

def main():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed_img = preprocess(gray_img)
        digits_positions = find_digits_positions(processed_img)
        recognized_digits = recognize_digits(digits_positions, processed_img)

        print("Recognized Digits:", recognized_digits)

        # Display the output frame
        for (x0, x1) in digits_positions:
            cv2.rectangle(frame, (x0, 0), (x1, frame.shape[0]), (0, 255, 0), 2)

        cv2.putText(frame, ' '.join(map(str, recognized_digits)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Detected Digits", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
