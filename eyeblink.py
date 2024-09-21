import numpy as np
import cv2
import time

# Initializing the face and eye cascade classifiers from xml files
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

# Variables to store the execution state
first_read = True
start_time_p1 = 0
start_time_p2 = 0
blink_time_p1 = None
blink_time_p2 = None
p1_lost = False
p2_lost = False

# Start video capture
cap = cv2.VideoCapture(0)
ret, img = cap.read()

# Define helper function to display text
def display_text(img, text, x, y, color=(255, 255, 255), size=2):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, size, color, 2)

# Split the screen vertically for two players
while ret:
    ret, img = cap.read()
    
    # Get the dimensions of the frame
    height, width, _ = img.shape
    middle = width // 2

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 1, 1)

    # Draw a black line to clearly divide the screen into two halves
    cv2.line(img, (middle, 0), (middle, height), (0, 0, 0), 5)

    # Split the frame for Player 1 (left side) and Player 2 (right side)
    img_p1 = img[:, :middle]
    img_p2 = img[:, middle:]

    # Detect face and eyes for Player 1
    faces_p1 = face_cascade.detectMultiScale(gray[:, :middle], 1.3, 5, minSize=(200, 200))
    player1_ready = len(faces_p1) > 0  # Check if Player 1 face is detected
    if player1_ready and not p1_lost:
        for (x, y, w, h) in faces_p1:
            img_p1 = cv2.rectangle(img_p1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_face = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_face, 1.3, 5, minSize=(50, 50))

            # Check if eyes are not detected (blink)
            if len(eyes) < 2:
                p1_lost = True
                blink_time_p1 = time.time()  # Record the blink time
                display_text(img, "Loser", 50, y + h + 30, (0, 0, 255), 3)

    # Detect face and eyes for Player 2
    faces_p2 = face_cascade.detectMultiScale(gray[:, middle:], 1.3, 5, minSize=(200, 200))
    player2_ready = len(faces_p2) > 0  # Check if Player 2 face is detected
    if player2_ready and not p2_lost:
        for (x, y, w, h) in faces_p2:
            img_p2 = cv2.rectangle(img_p2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_face = gray[y:y+h, x+middle:x+middle+w]
            eyes = eye_cascade.detectMultiScale(roi_face, 1.3, 5, minSize=(50, 50))

            # Check if eyes are not detected (blink)
            if len(eyes) < 2:
                p2_lost = True
                blink_time_p2 = time.time()  # Record the blink time
                display_text(img, "Loser", middle + 50, y + h + 30, (0, 0, 255), 3)

    # If one player loses, declare the other as the winner
    if p1_lost and not p2_lost:
        display_text(img, "Winner", middle + 50, height // 2, (0, 255, 0), 3)
    elif p2_lost and not p1_lost:
        display_text(img, "Winner", 50, height // 2, (0, 255, 0), 3)
    elif p1_lost and p2_lost:
        # Declare the winner based on who blinked **later**
        if blink_time_p1 > blink_time_p2:
            display_text(img, "Player 1 is the Winner!", width // 3, height // 2, (0, 255, 0), 3)
        else:
            display_text(img, "Player 2 is the Winner!", width // 3, height // 2, (0, 255, 0), 3)

    # Continue game logic when players press 's' and detection starts
    if not first_read and not p1_lost and not p2_lost:
        # Handle Player 1's score in seconds
        if player1_ready and not p1_lost:
            elapsed_time_p1 = round((time.time() - start_time_p1), 2) if start_time_p1 else 0
            display_text(img, f"P1: {elapsed_time_p1} sec", 50, height - 50)

        # Handle Player 2's score in seconds
        if player2_ready and not p2_lost:
            elapsed_time_p2 = round((time.time() - start_time_p2), 2) if start_time_p2 else 0
            display_text(img, f"P2: {elapsed_time_p2} sec", middle + 50, height - 50)

    # Display 'Press s to start' when both players are detected
    if player1_ready and player2_ready and first_read:
        display_text(img, "Press 's' to start", width // 3, height // 2, (0, 255, 0), 3)

    # Show the full frame with two sides
    cv2.imshow('Blinking Game', img)

    # Control the algorithm with keys
    a = cv2.waitKey(1)
    if a == ord('q'):
        break
    elif a == ord('s') and first_read and player1_ready and player2_ready:
        # Start the detection for both players when 's' is pressed
        first_read = False
        start_time_p1 = time.time()
        start_time_p2 = time.time()

cap.release()
cv2.destroyAllWindows()
