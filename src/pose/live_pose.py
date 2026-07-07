"""
SpotMe Phase 1: Live Pose Estimation
=====================================
This script opens your webcam, runs MediaPipe Pose on every frame,
and draws a skeleton on top of the video feed in real time.

HOW IT WORKS:
1. OpenCV grabs a frame from your webcam (just a numpy array of pixels)
2. We convert BGR -> RGB (OpenCV uses BGR, MediaPipe expects RGB)
3. MediaPipe's Pose model processes the frame and returns 33 landmarks
4. Each landmark has: x (0-1), y (0-1), z (depth), visibility (confidence)
5. We draw circles at each landmark and lines connecting them = skeleton
6. Display the annotated frame

Run: python src/pose/live_pose.py
Quit: press 'q'
"""

import cv2
import mediapipe as mp
import numpy as np

# ──────────────────────────────────────────────
# SETUP: Initialize MediaPipe Pose
# ──────────────────────────────────────────────
#
# mp.solutions.pose is the Pose module — it contains a pre-trained
# neural network (BlazePose) that was trained on thousands of images
# of human bodies. You don't need to train anything.
#
# Key parameters:
#   min_detection_confidence: how sure the model needs to be to say
#       "I see a person" (0.0 to 1.0). Lower = more detections but
#       more false positives. 0.5 is a good default.
#
#   min_tracking_confidence: once a person is detected, how sure the
#       model needs to be to keep tracking them frame-to-frame.
#       Higher = smoother tracking but might lose you if you move fast.
#
#   model_complexity: 0 (lite/fast), 1 (full/balanced), 2 (heavy/accurate)
#       We use 1 for a good balance of speed and accuracy.

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,
)

# ──────────────────────────────────────────────
# WEBCAM: Open a video capture stream
# ──────────────────────────────────────────────
#
# cv2.VideoCapture(0) opens the default webcam.
# If you have multiple cameras, try 1, 2, etc.
# You can also pass a video file path instead of 0
# to test with pre-recorded footage.

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Can't open webcam. Make sure it's connected.")
    print("If testing without a webcam, change VideoCapture(0) to a video file path.")
    exit(1)

# Set resolution (optional — higher = more detail but slower)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("SpotMe Phase 1 — Live Pose Estimation")
print("Press 'q' to quit")
print("--------------------------------------")

# ──────────────────────────────────────────────
# LANDMARK INDICES — the ones we care about for lifting
# ──────────────────────────────────────────────
# MediaPipe gives us 33 landmarks. Here are the key ones:
#
#   0  = nose
#   11 = left shoulder      12 = right shoulder
#   13 = left elbow         14 = right elbow
#   15 = left wrist         16 = right wrist
#   23 = left hip           24 = right hip
#   25 = left knee          26 = right knee
#   27 = left ankle         28 = right ankle
#
# Full list: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

# Landmarks we'll highlight (the ones relevant to exercise form)
KEY_LANDMARKS = {
    11: "L Shoulder", 12: "R Shoulder",
    13: "L Elbow",    14: "R Elbow",
    15: "L Wrist",    16: "R Wrist",
    23: "L Hip",      24: "R Hip",
    25: "L Knee",     26: "R Knee",
    27: "L Ankle",    28: "R Ankle",
}

# ──────────────────────────────────────────────
# MAIN LOOP: Process each frame
# ──────────────────────────────────────────────

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame — camera might have disconnected")
        break

    # Flip horizontally so it mirrors you (more intuitive)
    frame = cv2.flip(frame, 1)

    # ── Step 1: Convert BGR to RGB ──
    # OpenCV reads images in BGR (Blue, Green, Red) order.
    # MediaPipe expects RGB. This is a common gotcha — if you
    # forget this, the model still "works" but accuracy drops
    # because the color channels are swapped.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ── Step 2: Run pose estimation ──
    # This is where the magic happens. The pre-trained BlazePose
    # neural network processes the frame and returns landmark data.
    #
    # Setting writeable=False before processing is a performance trick:
    # it tells numpy not to allow writes to the array, which lets
    # MediaPipe avoid copying the data internally.
    rgb_frame.flags.writeable = False
    results = pose.process(rgb_frame)
    rgb_frame.flags.writeable = True

    # ── Step 3: Draw the skeleton ──
    if results.pose_landmarks:
        # Draw the full skeleton (all connections)
        # This uses MediaPipe's built-in drawing — it knows which
        # landmarks connect to which (shoulder→elbow→wrist, etc.)
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )

        # Draw labels on key landmarks
        h, w, _ = frame.shape  # frame dimensions in pixels
        for idx, name in KEY_LANDMARKS.items():
            lm = results.pose_landmarks.landmark[idx]

            # Landmarks come as normalized coords (0.0 to 1.0).
            # Multiply by frame dimensions to get pixel positions.
            px = int(lm.x * w)
            py = int(lm.y * h)

            # Only label if the landmark is visible enough
            # (visibility is the model's confidence that this
            # joint is actually visible in the frame, not occluded)
            if lm.visibility > 0.5:
                cv2.putText(
                    frame, name, (px + 10, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA
                )

        # ── Step 4: Print some landmark data (for learning) ──
        # Grab left shoulder and left elbow to see the raw numbers
        l_shoulder = results.pose_landmarks.landmark[11]
        l_elbow = results.pose_landmarks.landmark[13]
        print(
            f"\rL Shoulder: ({l_shoulder.x:.2f}, {l_shoulder.y:.2f}) "
            f"vis={l_shoulder.visibility:.2f}  |  "
            f"L Elbow: ({l_elbow.x:.2f}, {l_elbow.y:.2f}) "
            f"vis={l_elbow.visibility:.2f}",
            end=""
        )
    else:
        # No person detected in frame
        cv2.putText(
            frame, "No person detected — step into frame",
            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 0, 255), 2, cv2.LINE_AA
        )

    # ── Step 5: Display the frame ──
    cv2.imshow("SpotMe — Phase 1", frame)

    # Wait 1ms for a key press. If 'q' is pressed, quit.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ──────────────────────────────────────────────
# CLEANUP
# ──────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
pose.close()
print("\nDone!")
