# 🏋️ Fitness AI Trainer - Automatic Exercise Recognition and Counting

An intelligent fitness application powered by **Computer Vision** and **Deep Learning** that automatically recognizes exercises and counts repetitions in real-time. Using advanced pose estimation and BiLSTM neural networks, this application provides accurate workout tracking through an intuitive web interface.

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## ✨ Features

- **Real-Time Exercise Recognition** - Automatically identify exercises from video or webcam input
- **Accurate Rep Counting** - Precise repetition counting using pose estimation and deep learning
- **Multiple Exercise Support** - Recognize various exercises including squats, push-ups, shoulder presses, and more
- **Form Analysis** - Get insights on exercise form and posture during workouts
- **Multi-Mode Operation** - Video analysis, live webcam mode, and auto-classification features
- **User-Friendly Interface** - Built with Streamlit for seamless interaction

## 🎯 How It Works

1. **Pose Estimation**: Uses MediaPipe to extract body landmarks from video frames
2. **Feature Extraction**: Calculates angles and distances between body joints
3. **Exercise Classification**: BiLSTM neural network classifies exercises based on movement patterns
4. **Repetition Counting**: Analyzes pose sequences to count completed repetitions
5. **Real-Time Feedback**: Provides immediate feedback on exercise performance

## 📚 Research

This project is based on the research paper:
> **"Real-Time Fitness Exercise Classification and Counting from Video Frames"**
> 
> [Read the paper on arXiv](https://arxiv.org/abs/2411.11548)

### Dataset
Training dataset available on Kaggle: [Real-Time Exercise Recognition Dataset](https://www.kaggle.com/datasets/riccardoriccio/real-time-exercise-recognition-dataset)

## 🎬 Demo

Watch the Fitness AI Trainer in action:
[![Watch the video](https://img.youtube.com/vi/GPmDPB1bSmc/hqdefault.jpg)](https://www.youtube.com/watch?v=GPmDPB1bSmc)

---

## 📋 Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Model Architecture](#model-architecture)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## ⚠️ Important Notes

1. **Model Variant**: The current repository uses the "BiLSTM Invariant" model, which uses angles and normalized distances without raw coordinates. For better performance, see `train_bidirectionallstm.py` which uses mixed features (coordinates + angles).

2. **Demo Videos**: Only `shoulder_press_form.mp4` is included. Additional instructional videos should be added to the `videos/` directory for full functionality.

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.7+** installed on your system
- **Git** for version control
- **Webcam** (optional, for real-time mode)
- **~2GB free disk space** for models and dependencies

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Fitness-AI-Trainer.git
   cd Fitness-AI-Trainer
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

**Run the main application:**
```bash
streamlit run main.py
```

The application will open in your default browser at `http://localhost:8501`

**Available Modes:**
- **Video Analysis**: Upload recorded videos to analyze exercise performance
- **Webcam Mode**: Real-time exercise tracking using your webcam
- **Auto Classify Mode**: Automatic exercise detection and counting

---

## 📁 Project Structure
```
Fitness-AI-Trainer/
├── main.py                                          # Main Streamlit application
├── ExerciseAiTrainer.py                            # Exercise-specific pose logic
├── AiTrainer_utils.py                              # Utility functions
├── PoseModule2.py                                  # MediaPipe pose estimation
├── extract_features.py                             # Feature extraction from videos
├── create_sequence_of_features.py                  # Dataset sequence generation
├── train_bidirectionallstm.py                      # Model training script
├── requirements.txt                                # Python dependencies
├── environment.yml                                 # Conda environment file
├── packages.txt                                    # System packages
├── final_forthesis_bidirectionallstm_and_encoders_exercise_classifier_model.h5  # Pre-trained model
├── thesis_bidirectionallstm_label_encoder.pkl      # Label encoder
├── thesis_bidirectionallstm_scaler.pkl             # Feature scaler
├── shoulder_press_form.mp4                         # Sample exercise video
└── README.md                                       # This file
```

### Key Files Explained

| File | Purpose |
|------|---------|
| `main.py` | Entry point for Streamlit application with UI/navigation |
| `PoseModule2.py` | Handles body pose detection using MediaPipe |
| `ExerciseAiTrainer.py` | Exercise-specific logic for form analysis |
| `extract_features.py` | Extracts landmarks and angles from video frames |
| `train_bidirectionallstm.py` | Training script for BiLSTM classifier models |

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.7+** | Core programming language |
| **TensorFlow/Keras** | Deep learning framework |
| **MediaPipe** | Pose estimation and landmark detection |
| **BiLSTM** | Exercise classification neural network |
| **Streamlit** | Web application framework |
| **NumPy/Pandas** | Data processing and analysis |
| **OpenCV** | Computer vision operations |

---

## 🧠 Model Architecture

### BiLSTM Invariant Classifier
- **Input**: Sequences of 30 frames with angle and distance features
- **Architecture**: Bidirectional LSTM layers for temporal sequence analysis
- **Output**: Exercise class prediction
- **Training Data**: Real-world exercise videos from Kaggle dataset

### Feature Set
- **Joint Angles**: Calculated between connected body joints
- **Normalized Distances**: Distance ratios for invariant representation
- **Temporal Context**: 30-frame sequences capture movement dynamics

### Exercise Classifier

The exercise classifier is built using a combination of real and synthetic datasets:

- **Kaggle Workout Dataset**: Real-world exercise videos
- **InfiniteRep Dataset**: Synthetic videos of avatars performing exercises
- **Similar Dataset**: Videos sourced from online to cover diverse exercise variations

The classification model employs LSTM and BiLSTM networks to process body landmarks and classify exercises based on joint angles and movement patterns. The model was optimized using accuracy, precision, recall, and F1-score metrics.

### Repetition Counting

Repetition counting is implemented in two modes:

1. **Manual Mode**: Users manually select the exercise, and repetitions are counted using angle-based thresholds
2. **Automatic Mode**: A BiLSTM model classifies exercises and applies counting logic based on identified body angles. The system tracks "up" and "down" movements to ensure accurate repetition counting

---

## 📊 Performance

The model achieves high accuracy on standard exercises:
- **Accuracy**: >95% on test set
- **Inference Speed**: Real-time performance on standard hardware
- **Supported Exercises**: Squats, push-ups, shoulder presses, and more

*For detailed performance metrics, refer to the research paper.*

---

## 🚨 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Module not found" error | Ensure virtual environment is activated and dependencies installed: `pip install -r requirements.txt` |
| Webcam not detected | Check camera permissions and ensure camera is not in use by another application |
| Streamlit not starting | Try running with: `streamlit run main.py --logger.level=debug` |
| Low accuracy detection | Ensure good lighting and camera positioning. Model performs best with clear, front-facing views |

---

## 📈 Future Enhancements

- [ ] Support for more exercise types
- [ ] Mobile app for iOS and Android
- [ ] Advanced form correction with real-time alerts
- [ ] Integration with fitness tracking apps (Apple Health, Google Fit)
- [ ] Multi-person exercise tracking
- [ ] Offline mode for privacy-focused users
- [ ] Customizable exercise library
- [ ] Performance analytics and workout history

---

## 🤝 Contributing

Contributions are welcome! Here's how to contribute:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/YourFeature`
3. **Commit changes**: `git commit -m 'Add YourFeature'`
4. **Push to branch**: `git push origin feature/YourFeature`
5. **Submit a Pull Request**

Please ensure your code follows PEP 8 standards and includes appropriate documentation.

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Original Author**: Riccardo Riccio
- **Research Paper**: "Real-Time Fitness Exercise Classification and Counting from Video Frames"
- **MediaPipe**: Google's powerful pose estimation framework
- **Dataset**: Kaggle Real-Time Exercise Recognition Dataset

---

## 📞 Contact & Support

For questions, suggestions, or collaboration opportunities:

- **Email**: riccardopersonalmail@gmail.com
- **LinkedIn**: [Riccardo Riccio](https://www.linkedin.com/in/riccardo-riccio-bb7163296/)
- **GitHub Issues**: [Report bugs or request features](../../issues)

---

## ⭐ If this project was helpful, please consider giving it a star!

**Happy training! 💪**
