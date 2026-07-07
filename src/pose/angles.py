"""
SpotMe Phase 2: Joint Angle Extraction
=======================================
This module calculates the angle at any joint given three landmarks.

THE CORE IDEA:
    A joint angle is defined by three points. The middle point is the
    "vertex" — where the angle actually lives. The other two points
    define the two "arms" of the angle.

    Example: Knee angle
        Point A = Hip (landmark 23)
        Point B = Knee (landmark 25)  ← vertex
        Point C = Ankle (landmark 27)

    The angle at B tells us how bent the knee is:
        180° = leg fully straight (standing)
        90°  = knee at right angle (parallel squat)
        <90° = deep squat (ass to grass)

THE MATH:
    1. Build two vectors from the vertex to each outer point
       Vector BA = A - B  (knee → hip direction)
       Vector BC = C - B  (knee → ankle direction)

    2. Dot product tells us how aligned the vectors are
       dot = BA.x * BC.x + BA.y * BC.y
       - If vectors point same direction: dot is large positive
       - If vectors are perpendicular: dot is ~0
       - If vectors point opposite ways: dot is large negative

    3. Divide by the product of their lengths (magnitudes)
       This "normalizes" the dot product to a range of -1 to 1

    4. arccos converts that normalized value to an angle
       arccos(1)  = 0°   (vectors identical)
       arccos(0)  = 90°  (vectors perpendicular)
       arccos(-1) = 180° (vectors opposite)

    That's it. Same formula for every joint in the body.
"""

import numpy as np


def calculate_angle(a, b, c):
    """
    Calculate the angle at point B formed by points A, B, C.

    Parameters:
        a: (x, y) or (x, y, z) — first outer point (e.g., hip)
        b: (x, y) or (x, y, z) — vertex point (e.g., knee)
        c: (x, y) or (x, y, z) — second outer point (e.g., ankle)

    Returns:
        Angle in degrees (0-180)

    Why numpy arrays?
        We convert to numpy arrays so we can do vector subtraction
        in one line (a - b) instead of manually subtracting each
        coordinate. It's cleaner and faster.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Step 1: Build vectors from the vertex (B) to each outer point
    # This is just subtraction: "which direction and how far is A from B?"
    ba = a - b  # Vector from knee to hip
    bc = c - b  # Vector from knee to ankle

    # Step 2: Dot product
    # np.dot multiplies corresponding components and sums them:
    # dot = ba[0]*bc[0] + ba[1]*bc[1] (+ ba[2]*bc[2] if 3D)
    #
    # Intuition: if both vectors point roughly the same way,
    # their components have the same signs, so the products are
    # positive and the sum is large. If they're perpendicular,
    # what's positive in one is zero in the other, so the sum is ~0.
    cosine_angle = np.dot(ba, bc)

    # Step 3: Divide by magnitudes (lengths of both vectors)
    # np.linalg.norm computes sqrt(x² + y²) — the Pythagorean length
    #
    # We need this because the dot product is affected by vector length.
    # A long vector dotted with another long vector gives a big number
    # even if the angle is wide. Dividing by the lengths removes the
    # length effect, leaving only the angle information.
    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)

    # Safety check: if either vector has zero length (two landmarks
    # are on top of each other), we can't compute an angle.
    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0.0

    cosine_angle = cosine_angle / (magnitude_ba * magnitude_bc)

    # Clamp to [-1, 1] to handle floating point errors
    # (arccos of 1.0000001 would crash without this)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    # Step 4: arccos converts the normalized dot product to radians,
    # then we convert radians to degrees (multiply by 180/π)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


# ──────────────────────────────────────────────
# LANDMARK TRIPLETS: which three points define each joint angle
# ──────────────────────────────────────────────
#
# Each tuple is (point_A, vertex_B, point_C)
# The vertex is always the joint we're measuring the angle AT.
#
# MediaPipe landmark indices:
#   11/12 = L/R shoulder
#   13/14 = L/R elbow
#   15/16 = L/R wrist
#   23/24 = L/R hip
#   25/26 = L/R knee
#   27/28 = L/R ankle

ANGLE_DEFINITIONS = {
    # ── Lower body (squat, deadlift, lunge) ──
    "left_knee":     (23, 25, 27),  # hip → KNEE → ankle
    "right_knee":    (24, 26, 28),
    "left_hip":      (25, 23, 11),  # knee → HIP → shoulder
    "right_hip":     (26, 24, 12),

    # ── Upper body (bench, OHP, curl) ──
    "left_elbow":    (11, 13, 15),  # shoulder → ELBOW → wrist
    "right_elbow":   (12, 14, 16),
    "left_shoulder":  (13, 11, 23),  # elbow → SHOULDER → hip
    "right_shoulder": (14, 12, 24),
}


def extract_all_angles(landmarks):
    """
    Given MediaPipe landmarks, extract all defined joint angles.

    Parameters:
        landmarks: MediaPipe pose_landmarks.landmark list
                   Each landmark has .x, .y, .z, .visibility

    Returns:
        Dictionary of {angle_name: angle_in_degrees}

    WHY WE USE (x, y) NOT (x, y, z):
        MediaPipe's z coordinate is relative depth from the camera,
        not a true 3D position. It's useful for some things but
        noisy for angle calculation. Using just x,y gives more
        stable angles for exercises performed facing the camera.
        We'll revisit this in Phase 3 when we use the full 3D
        skeleton graph.
    """
    angles = {}

    for name, (a_idx, b_idx, c_idx) in ANGLE_DEFINITIONS.items():
        # Get the three landmarks
        a = landmarks[a_idx]
        b = landmarks[b_idx]
        c = landmarks[c_idx]

        # Check visibility — all three points need to be visible
        # for the angle to be meaningful. If any joint is occluded,
        # the coordinates are just MediaPipe's best guess and the
        # angle would be noisy/wrong.
        min_visibility = min(a.visibility, b.visibility, c.visibility)
        if min_visibility < 0.5:
            angles[name] = None  # Not confident enough
            continue

        # Calculate the angle using (x, y) coordinates
        angle = calculate_angle(
            (a.x, a.y),
            (b.x, b.y),
            (c.x, c.y),
        )
        angles[name] = round(angle, 1)

    return angles


def get_body_side_angles(angles):
    """
    Returns the angles for whichever side of the body is more visible.

    WHY THIS MATTERS:
        When you're squatting facing the camera at an angle, one side
        of your body is more visible than the other. The more visible
        side has more accurate angles. This function picks the side
        where more angles are successfully detected.

        In Phase 3, we'll use both sides for the graph neural network.
        For now, using the better side gives cleaner data.
    """
    left_count = sum(1 for k, v in angles.items() if "left" in k and v is not None)
    right_count = sum(1 for k, v in angles.items() if "right" in k and v is not None)

    if left_count >= right_count:
        return {
            "knee": angles.get("left_knee"),
            "hip": angles.get("left_hip"),
            "elbow": angles.get("left_elbow"),
            "shoulder": angles.get("left_shoulder"),
        }
    else:
        return {
            "knee": angles.get("right_knee"),
            "hip": angles.get("right_hip"),
            "elbow": angles.get("right_elbow"),
            "shoulder": angles.get("right_shoulder"),
        }
