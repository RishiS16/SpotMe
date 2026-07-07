# SpotMe

**Real-time exercise form analysis and coaching powered by graph neural networks, contrastive learning, and biomechanical modeling.**

Point your camera at yourself while lifting. SpotMe tracks your skeleton, classifies form quality using spatial-temporal graph convolutions, estimates joint torques for injury risk, and delivers real-time voice coaching powered by Claude AI.

> *Like having a sports science lab in your pocket.*

---

## How it works

1. **Pose estimation** — MediaPipe tracks 33 body landmarks at 30fps from a single camera
2. **Skeleton graph construction** — The body is modeled as a graph (joints = nodes, bones = edges), preserving spatial structure
3. **Joint angle extraction** — Calculates knee, hip, elbow, and spine angles using vector math on 3D landmark coordinates
4. **ST-GCN form classification** — A Spatial-Temporal Graph Convolutional Network learns body part relationships (spatial) and movement patterns across frames (temporal) to score form quality
5. **Contrastive pre-training** — Self-supervised learning on unlabeled movement sequences using NT-Xent loss, reducing labeled data requirements by ~95%
6. **Biomechanical torque estimation** — Inverse dynamics estimates rotational force on each joint using pose data and anthropometric segment models, enabling injury risk scoring
7. **Claude AI coaching** — Claude API receives joint angles, form scores, and torque data, then generates specific coaching cues delivered via Web Speech API during live sets
8. **Progress tracking** — Every session logs to a dashboard showing form scores, torque trends, and volume over time

---

## Supported exercises

| Exercise | Key landmarks | Form checks |
|----------|--------------|-------------|
| Barbell squat | Hip, knee, ankle | Depth, knee tracking, back angle, valgus |
| Bench press | Shoulder, elbow, wrist | Bar path, elbow flare, arch, lockout |
| Deadlift | Hip, knee, spine | Back rounding, hip hinge, lockout, bar drift |
| Overhead press | Shoulder, elbow, wrist | Bar path, lean back, lockout, flare |
| Bicep curl | Shoulder, elbow, wrist | Elbow drift, body swing, full ROM |

---

## Tech stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Pose estimation | MediaPipe Pose, OpenCV | 33-landmark body tracking at 30fps |
| Graph ML | PyTorch Geometric, ST-GCN | Skeleton graph convolution for form classification |
| Self-supervised learning | Contrastive learning (NT-Xent) | Pre-training on unlabeled movement data |
| Biomechanics | Inverse dynamics, SciPy | Joint torque and injury risk estimation |
| Training | PyTorch, CUDA, mixed precision (AMP) | GPU-accelerated model training |
| Experiment tracking | Weights & Biases, MLflow | Hyperparameter logging and metric visualization |
| Model deployment | ONNX Runtime | Optimized inference for real-time prediction |
| AI coaching | Claude API (Anthropic) | Context-aware coaching cue generation |
| Voice output | Web Speech API | Hands-free audio coaching during sets |
| Backend | FastAPI, PostgreSQL, Redis | REST API, session storage, caching |
| Real-time | WebRTC, WebSockets | Camera streaming and live data push |
| Frontend | React, TailwindCSS, Recharts | Progress dashboard and session viewer |
| Infrastructure | Docker, AWS ECS, GitHub Actions | Containerized deployment and CI/CD |
| Training hardware | NVIDIA DGX Spark / Google Colab T4 | GPU training environment |

---

## What makes this different

Most exercise form projects use a CNN or LSTM on flattened joint coordinates. SpotMe takes a research-grade approach:

**ST-GCN (Spatial-Temporal Graph Convolutional Network)** — Models the body as a graph, preserving skeletal topology. Spatial convolutions learn that shoulder-elbow-wrist is a kinetic chain. Temporal convolutions learn how the movement unfolds across frames. This is the state-of-the-art architecture for skeleton-based action recognition from top CV conferences.

**Contrastive self-supervised pre-training** — Instead of manually labeling thousands of reps, the model learns a movement embedding space where similar form clusters together, using custom skeleton augmentations (joint noise, temporal crop, spatial dropout, speed perturbation). Fine-tuning requires only ~50 labeled examples per class.

**Biomechanical torque estimation** — Goes beyond "good/bad" classification to estimate actual rotational forces on each joint using inverse dynamics and anthropometric segment models. High torques at extreme angles flag injury risk. This is what sports science labs do with $100K motion capture rigs — approximated here from a single camera.

---

## Quick start

```bash
# Clone and install
git clone https://github.com/RishiS16/SpotMe.git
cd SpotMe
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run Phase 1: Live pose estimation
python src/pose/live_pose.py
```

---

## Project structure

```
spotme/
├── src/
│   ├── pose/              # MediaPipe pose estimation
│   │   ├── live_pose.py   # Webcam skeleton overlay
│   │   ├── angles.py      # Joint angle extraction
│   │   └── rep_counter.py # Rep detection logic
│   ├── graph/             # Skeleton graph construction
│   │   ├── skeleton.py    # Adjacency matrix + graph builder
│   │   └── augment.py     # Contrastive augmentations
│   ├── classifier/        # ST-GCN model
│   │   ├── stgcn.py       # Model architecture
│   │   ├── contrastive.py # NT-Xent pre-training
│   │   └── train.py       # Training loop (CUDA + AMP)
│   ├── biomechanics/      # Torque estimation
│   │   ├── dynamics.py    # Inverse dynamics equations
│   │   └── risk.py        # Injury risk scoring
│   ├── coaching/          # Claude integration
│   │   ├── prompt.py      # Prompt templates
│   │   └── voice.py       # Web Speech API output
│   ├── api/               # FastAPI backend
│   │   ├── main.py        # App entry point
│   │   ├── routes.py      # API endpoints
│   │   ├── models.py      # SQLAlchemy ORM models
│   │   └── ws.py          # WebSocket handlers
│   └── dashboard/         # React frontend
├── data/
│   ├── raw/               # Recorded video data
│   ├── processed/         # Extracted landmarks
│   └── models/            # Trained weights (.pt, .onnx)
├── notebooks/             # Exploration and analysis
├── tests/                 # Unit and integration tests
├── config/                # Configuration files
├── Dockerfile
├── .github/workflows/     # CI/CD pipeline
├── requirements.txt
└── README.md
```

---

## Development phases

- [x] Phase 1: Live pose estimation with MediaPipe
- [ ] Phase 2: Joint angle extraction and rep counting
- [ ] Phase 3: ST-GCN form classifier with contrastive pre-training
- [ ] Phase 4: Biomechanical analysis + Claude coaching integration
- [ ] Phase 5: FastAPI backend + React dashboard
- [ ] Phase 6: Docker, CI/CD, and deployment

---

## Research directions

This project serves as a platform for several publishable research threads:

- **Monocular torque estimation vs. lab ground truth** — Validating phone-based biomechanical analysis against Vicon motion capture systems
- **Skeleton augmentations for contrastive learning** — Systematic evaluation of augmentation strategies for self-supervised skeleton representation learning
- **Few-shot exercise recognition for physical therapy** — Meta-learning systems that learn new PT exercises from 3-5 therapist demonstrations
- **Body-type bias in pose-based assessment** — Fairness analysis of how limb proportions affect form classification accuracy
- **LLM-as-coach evaluation** — Comparing Claude's biomechanical reasoning against certified strength coaches

---

## License

MIT
