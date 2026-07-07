# SpotMe — AI Personal Trainer

Real-time exercise form analysis and coaching powered by computer vision and AI.

Point your camera at yourself while lifting. SpotMe counts your reps, scores your form, and coaches you through voice feedback — like having a personal trainer in your pocket.

## How it works

1. **Pose estimation** — MediaPipe tracks 33 body landmarks at 30fps
2. **Joint angle extraction** — Calculates knee, hip, elbow, and spine angles per frame
3. **Form classification** — Custom PyTorch model scores form quality per rep
4. **AI coaching** — Claude API generates specific coaching cues from joint angle data
5. **Voice feedback** — Web Speech API delivers cues hands-free during your set
6. **Progress tracking** — Every session logs to a dashboard showing form scores and volume over time

## Supported exercises

- Squat (barbell, goblet, bodyweight)
- Bench press
- Deadlift (conventional, sumo)
- Overhead press
- Bicep curl

## Tech stack

| Layer | Tech |
|-------|------|
| Pose estimation | MediaPipe Pose, OpenCV |
| Form classifier | PyTorch CNN |
| AI coaching | Claude API (Anthropic) |
| Voice output | Web Speech API |
| Backend | FastAPI, PostgreSQL, Redis |
| Frontend | React, TailwindCSS, Recharts |
| Real-time | WebRTC, WebSockets |
| Infra | Docker, AWS, GitHub Actions |

## Quick start

```bash
# Clone and install
git clone https://github.com/yourusername/spotme.git
cd spotme
pip install -r requirements.txt

# Run Phase 1: Live pose estimation
python src/pose/live_pose.py
```

## Project structure

```
spotme/
├── src/
│   ├── pose/           # Pose estimation and landmark extraction
│   ├── classifier/     # PyTorch form quality classifier
│   ├── coaching/       # Claude API integration for coaching cues
│   ├── api/            # FastAPI backend
│   └── dashboard/      # React frontend
├── data/
│   ├── raw/            # Raw video/frame data
│   ├── processed/      # Extracted landmarks and features
│   └── models/         # Trained model weights
├── notebooks/          # Jupyter notebooks for exploration
├── tests/              # Unit and integration tests
└── config/             # Configuration files
```

## Development phases

- [x] Phase 1: Live pose estimation with MediaPipe
- [ ] Phase 2: Joint angle extraction and math
- [ ] Phase 3: Form classifier (PyTorch)
- [ ] Phase 4: Claude coaching integration
- [ ] Phase 5: FastAPI backend + React dashboard
- [ ] Phase 6: Polish and deployment

## License

MIT
