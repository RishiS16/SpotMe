"""
SpotMe Phase 2: Rep Counter
============================
Detects and counts exercise repetitions from joint angle data.

THE CORE IDEA:
    A rep is a cycle in the joint angle over time. Take a squat:

    Time:   0s    1s    2s    3s    4s    5s
    Angle:  170°  140°  95°   90°   130°  170°
            ^                         ^
            standing                  standing
                        v
                     bottom of squat

    That's one rep: angle goes DOWN past a threshold, then comes
    back UP past a threshold.

    We track this with a simple state machine:

        State: "UP" (standing)
             │
             ▼ angle drops below DOWN_THRESHOLD
        State: "DOWN" (in the hole)
             │
             ▼ angle rises above UP_THRESHOLD
        State: "UP" + rep_count += 1

    That's it. Two thresholds, two states, count the transitions.

WHY TWO THRESHOLDS INSTEAD OF ONE:
    If we used a single threshold (say 130°), the angle would
    rapidly oscillate across it at the bottom of the movement,
    counting multiple phantom reps. Using separate UP and DOWN
    thresholds creates a "dead zone" that prevents this.

    This is called HYSTERESIS — the same technique used in
    thermostats (turn heater on at 68°, off at 72°, not
    toggling at exactly 70°).
"""


class RepCounter:
    """
    Counts reps for a single exercise based on one joint angle.

    Parameters:
        down_threshold: angle (degrees) below which we consider
                        the person to be in the "down" position
        up_threshold:   angle (degrees) above which we consider
                        the person to be back in the "up" position

    Example for squat:
        counter = RepCounter(down_threshold=100, up_threshold=160)

        At the bottom of a squat, knee angle drops below 100° → state = DOWN
        As you stand back up, knee angle rises above 160° → state = UP, count +1
    """

    def __init__(self, down_threshold, up_threshold):
        self.down_threshold = down_threshold
        self.up_threshold = up_threshold
        self.state = "UP"     # Start assuming person is standing
        self.count = 0        # Total reps completed
        self.min_angle = 180  # Tracks the deepest angle in current rep
        self.rep_history = [] # Stores the min angle of each completed rep

    def update(self, angle):
        """
        Feed in a new angle reading. Returns True if a rep was just completed.

        This gets called every frame (~30 times per second). Most frames
        nothing interesting happens — the angle is just moving gradually.
        We only care about the moments it crosses our thresholds.

        Parameters:
            angle: current joint angle in degrees (or None if not visible)

        Returns:
            True if a rep was just completed on this frame, False otherwise
        """
        if angle is None:
            return False

        completed_rep = False

        # Track the minimum angle during the "down" phase
        # This tells us how deep the person went
        if angle < self.min_angle:
            self.min_angle = angle

        # State machine transition: UP → DOWN
        # "You've gone low enough, you're in the rep now"
        if self.state == "UP" and angle < self.down_threshold:
            self.state = "DOWN"

        # State machine transition: DOWN → UP
        # "You've come back up, that's one rep"
        elif self.state == "DOWN" and angle > self.up_threshold:
            self.state = "UP"
            self.count += 1
            completed_rep = True

            # Record how deep this rep went (useful for form analysis later)
            self.rep_history.append({
                "rep_number": self.count,
                "min_angle": round(self.min_angle, 1),
            })

            # Reset min angle tracking for next rep
            self.min_angle = 180

        return completed_rep

    def reset(self):
        """Reset the counter for a new set."""
        self.count = 0
        self.state = "UP"
        self.min_angle = 180
        self.rep_history = []


# ──────────────────────────────────────────────
# EXERCISE PRESETS
# ──────────────────────────────────────────────
# Each exercise uses a specific joint angle and thresholds.
#
# These thresholds are starting points — in Phase 3, the
# ST-GCN model will learn to detect reps from the full
# skeleton graph instead of relying on manual thresholds.
# But for now, thresholds work surprisingly well.
#
# HOW TO PICK THRESHOLDS:
#   1. Do the exercise in front of the camera
#   2. Watch the angle printout in the terminal
#   3. Note the angle at the bottom of the movement (→ down_threshold)
#   4. Note the angle when you're standing (→ up_threshold)
#   5. Add some margin so it doesn't trigger too early

EXERCISE_CONFIGS = {
    "squat": {
        "angle_name": "knee",        # Which angle to track
        "down_threshold": 100,        # Below this = bottom of squat
        "up_threshold": 160,          # Above this = standing
        "description": "Tracks knee angle. Full depth = ~90°",
    },
    "deadlift": {
        "angle_name": "hip",
        "down_threshold": 100,
        "up_threshold": 160,
        "description": "Tracks hip hinge angle. Bottom = ~90°",
    },
    "bicep_curl": {
        "angle_name": "elbow",
        "down_threshold": 50,         # Elbow fully bent at top of curl
        "up_threshold": 140,          # Arm almost straight at bottom
        "description": "Tracks elbow angle. Contracted = ~40°",
    },
    "overhead_press": {
        "angle_name": "elbow",
        "down_threshold": 80,
        "up_threshold": 160,
        "description": "Tracks elbow angle. Bottom = ~80°, lockout = ~170°",
    },
    "bench_press": {
        "angle_name": "elbow",
        "down_threshold": 80,
        "up_threshold": 155,
        "description": "Tracks elbow angle. Chest = ~80°, lockout = ~165°",
    },
}


def create_counter(exercise_name):
    """
    Create a RepCounter configured for a specific exercise.

    Parameters:
        exercise_name: one of the keys in EXERCISE_CONFIGS

    Returns:
        (RepCounter, config_dict)
    """
    if exercise_name not in EXERCISE_CONFIGS:
        raise ValueError(
            f"Unknown exercise '{exercise_name}'. "
            f"Available: {list(EXERCISE_CONFIGS.keys())}"
        )

    config = EXERCISE_CONFIGS[exercise_name]
    counter = RepCounter(
        down_threshold=config["down_threshold"],
        up_threshold=config["up_threshold"],
    )
    return counter, config
