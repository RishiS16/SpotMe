"""
SpotMe Phase 2: Live Pose with Angles + Rep Counter
=====================================================
Builds on Phase 1 by adding:
  - Real-time joint angle calculation and display
  - Automatic rep counting with hysteresis
  - Exercise selection

Run: python src/pose/live_pose_v2.py
     python src/pose/live_pose_v2.py --exercise squat
     python src/pose/live_pose_v2.py --exercise bicep_curl

Quit: press 'q'
Reset rep count: press 'r'
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import sys
import os

# Add the project root to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.pose.angles import extract_all_angles, get_body_side_angles
from src.pose.rep_counter import create_counter, EXERCISE_CONFIGS

# ──────────────────────────────────────────────
# ARGUMENT PARSING
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser(description="SpotMe — Live Form Analysis")
parser.add_argument(
    "--exercise", type=str, default="squat",
    choices=list(EXERCISE_CONFIGS.keys()),
    help="Exercise to track (default: squat)"
)
args = parser.parse_args()

# ──────────────────────────────────────────────
# SETUP
# ──────────────────────────────────────────────
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Can't open webcam.")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Create rep counter for selected exercise
counter, config = create_counter(args.exercise)
tracked_angle_name = config["angle_name"]

print(f"SpotMe Phase 2 — {args.exercise.replace('_', ' ').title()}")
print(f"Tracking: {tracked_angle_name} angle")
print(f"Down threshold: {config['down_threshold']}° | Up threshold: {config['up_threshold']}°")
print(f"{config['description']}")
print(f"Press 'q' to quit, 'r' to reset rep count")
print("─" * 50)


def draw_angle_arc(frame, center, angle, radius=40):
    """
    Draw a visual arc showing the joint angle on the video frame.

    This is purely cosmetic — it draws a colored arc at the joint
    so you can visually see the angle. Green = good range, yellow =
    getting low, red = very deep.

    Parameters:
        frame: the video frame (numpy array)
        center: (x, y) pixel position of the joint
        angle: the joint angle in degrees
        radius: size of the arc drawing
    """
    # Color based on angle: green (good) → yellow (caution) → red (deep)
    if angle > 140:
        color = (0, 200, 0)      # Green — upright/safe
    elif angle > 100:
        color = (0, 200, 200)    # Yellow — mid-range
    else:
        color = (0, 0, 200)      # Red — deep/extreme

    # Draw the angle value as text
    cv2.putText(
        frame, f"{int(angle)}",
        (center[0] - 20, center[1] - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
    )


# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # ── Run MediaPipe ──
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = pose.process(rgb_frame)
    rgb_frame.flags.writeable = True

    if results.pose_landmarks:
        # Draw skeleton
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )

        # ── Extract all joint angles ──
        # This calls our calculate_angle function for every joint
        # defined in ANGLE_DEFINITIONS (knee, hip, elbow, shoulder)
        all_angles = extract_all_angles(results.pose_landmarks.landmark)

        # Get the best side's angles
        side_angles = get_body_side_angles(all_angles)

        # ── Update rep counter ──
        # Feed the tracked angle into the state machine
        tracked_angle = side_angles.get(tracked_angle_name)
        just_completed = counter.update(tracked_angle)

        if just_completed:
            # A rep was just completed! Print the details
            last_rep = counter.rep_history[-1]
            print(
                f"  Rep {last_rep['rep_number']}: "
                f"min angle = {last_rep['min_angle']}°"
            )

        # ── Draw angles on the frame ──
        # Show the angle value at each visible joint
        landmark_positions = {
            "knee": (25, 26),      # left, right landmark indices
            "hip": (23, 24),
            "elbow": (13, 14),
            "shoulder": (11, 12),
        }

        for angle_name, (left_idx, right_idx) in landmark_positions.items():
            # Try left side first, then right
            angle_val = all_angles.get(f"left_{angle_name}") or all_angles.get(f"right_{angle_name}")
            idx = left_idx if all_angles.get(f"left_{angle_name}") else right_idx

            if angle_val is not None:
                lm = results.pose_landmarks.landmark[idx]
                px, py = int(lm.x * w), int(lm.y * h)
                draw_angle_arc(frame, (px, py), angle_val)

        # ── Draw HUD (heads-up display) ──

        # Exercise name
        cv2.putText(
            frame, f"{args.exercise.replace('_', ' ').upper()}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (255, 255, 255), 2, cv2.LINE_AA
        )

        # Rep count (big number)
        cv2.putText(
            frame, f"REPS: {counter.count}",
            (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
            (0, 255, 0), 3, cv2.LINE_AA
        )

        # Current state (UP/DOWN)
        state_color = (0, 255, 0) if counter.state == "UP" else (0, 165, 255)
        cv2.putText(
            frame, f"State: {counter.state}",
            (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            state_color, 2, cv2.LINE_AA
        )

        # Tracked angle value
        if tracked_angle is not None:
            cv2.putText(
                frame, f"{tracked_angle_name}: {tracked_angle:.0f} deg",
                (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA
            )

        # Rep flash — briefly flash the screen green on rep completion
        if just_completed:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 0), -1)
            frame = cv2.addWeighted(overlay, 0.1, frame, 0.9, 0)

    else:
        cv2.putText(
            frame, "No person detected — step into frame",
            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 0, 255), 2, cv2.LINE_AA
        )

    # Display
    cv2.imshow("SpotMe — Phase 2", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        counter.reset()
        print("Rep counter reset!")

# ──────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────
print(f"\n{'─' * 50}")
print(f"Session summary: {counter.count} reps of {args.exercise}")
if counter.rep_history:
    angles = [r["min_angle"] for r in counter.rep_history]
    print(f"  Depth range: {min(angles):.0f}° – {max(angles):.0f}°")
    print(f"  Average depth: {sum(angles)/len(angles):.0f}°")
    print(f"  Per-rep breakdown:")
    for rep in counter.rep_history:
        print(f"    Rep {rep['rep_number']}: {rep['min_angle']}°")
print(f"{'─' * 50}")

cap.release()
cv2.destroyAllWindows()
pose.close()
