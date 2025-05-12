import cv2
import numpy as np

# Initialize the camera
cap = cv2.VideoCapture(0)

# Background subtraction
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

# Kernel for morphological operations
kernel = np.ones((3, 3), np.uint8)

# Minimum area threshold for contour detection
min_contour_area = 10000

# Gesture recognition history
gesture_history = []

def detect_gesture(frame):
    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)
    
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to remove noise
    thresh = cv2.erode(thresh, kernel, iterations=2)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None, frame
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Filter out small contours
    if cv2.contourArea(largest_contour) < min_contour_area:
        return None, frame
    
    # Draw the contour
    cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
    
    # Convex hull and defects for hand gesture recognition
    hull = cv2.convexHull(largest_contour, returnPoints=False)
    defects = cv2.convexityDefects(largest_contour, hull)
    
    if defects is None:
        return None, frame
    
    # Count fingers based on convexity defects
    finger_count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(largest_contour[s][0])
        end = tuple(largest_contour[e][0])
        far = tuple(largest_contour[f][0])
        
        # Calculate angles between fingers
        a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c)) * 57
        
        # Ignore wide angles and points that are too close
        if angle <= 90 and d > 10000:
            finger_count += 1
            cv2.circle(frame, far, 4, [0, 0, 255], -1)
    
    # The finger count is usually defects + 1
    if finger_count > 0:
        finger_count += 1
    
    # Ensure finger count is between 0 and 5
    finger_count = min(5, max(0, finger_count))
    
    return finger_count, frame

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    
    # Detect gesture
    gesture, output_frame = detect_gesture(frame)
    
    if gesture is not None:
        gesture_history.append(gesture)
        if len(gesture_history) > 5:
            gesture_history.pop(0)
        
        # Get the most frequent gesture in history
        if len(gesture_history) > 0:
            final_gesture = max(set(gesture_history), key=gesture_history.count)
            cv2.putText(output_frame, f'Fingers: {final_gesture}', (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Show the output
    cv2.imshow('Gesture Recognition', output_frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
