# EKF-SLAM Simulation with Python

A visual and educational implementation of **Extended Kalman Filter (EKF) SLAM** in Python, designed to simulate a mobile robot navigating an unknown 2D environment, detecting landmarks using noisy sensors, and incrementally building a map.

![EKF SLAM Preview](https://github.com/EricChen0104/EKF_SLAM_PYTHON/blob/master/assets/Demo%20Video%20to%20GIF.gif)

## 🌟 Features

- ✅ Fully functional EKF prediction and update steps
- 📍 Data association with Mahalanobis distance threshold
- 📡 Landmark initialization with measurement uncertainty
- 🚗 Vehicle motion modeled with noisy velocity and steering
- 🧠 Visualization of:
  - True robot trajectory
  - EKF estimated trajectory
  - Detected landmarks
  - Uncertainty ellipses (covariance)
  - Detection range and visible connections
- 🎯 Supports goal navigation between multiple waypoints

## 📦 Dependencies

- Python 3.7+
- `numpy`
- `matplotlib`

Install them with:

```bash

pip install numpy matplotlib

```

▶️ Run the Simulation

```bash

python EKF_SLAM.py

```

This will launch a real-time matplotlib animation of a robot moving across a 30x30 grid, detecting 25 randomly placed obstacles and navigating through several predefined waypoints using EKF-SLAM.

## 🧠 Core Concepts

This simulation helps you understand:

- How the EKF updates both robot pose and landmark positions
- Handling noisy motion and sensor inputs
- Using Mahalanobis distance for data association
- Visualizing SLAM performance in real time

## 🚀 Applications

This is a great starting point for:

- Learning SLAM algorithms
- Robotics education and demos
- Integrating with more advanced simulators (e.g., MuJoCo, PyBullet)
- Developing autonomous navigation pipelines

## 📌 To-Do & Ideas

- Add saving & loading of maps
- Export logs for post-analysis
- Integrate real-world LIDAR data
- Extend to 3D SLAM

## 🤝 Contributing

Pull requests and forks are welcome! If you found this project helpful, feel free to give it a ⭐️ and share it.
