# Radar-Camera Fusion Object Detection System

This project demonstrates a sensor fusion system combining **Camera** (Computer Vision) and **Radar** data for robust object detection. It uses a Streamlit interface to visualize the fusion results in real-time (using a webcam and simulated radar data).

## Why Radar-Camera Fusion?

Sensor fusion combines data from multiple sensors to reduce uncertainties and create a more accurate perception of the environment.

*   **Camera Strengths**: High resolution, rich texture/color information, excellent for **Object Classification** (identifying *what* an object is).
*   **Camera Weaknesses**: Performance degrades in poor lighting (night, glare) or bad weather (rain, fog). Lacks direct depth and velocity measurement (requires estimation).
*   **Radar Strengths**: Direct measurement of **Range (Distance)** and **Velocity (Doppler)**. Works robustly in almost all weather conditions and lighting.
*   **Radar Weaknesses**: Low lateral resolution, poor object classification capabilities (hard to tell a pedestrian from a signpost based on radar alone).

**Fusion Benefit**: By combining these, we get the best of both worlds:
1.  **Robustness**: If the camera fails (e.g., blinding sun), radar still detects the obstacle.
2.  **Accuracy**: Camera provides the class label ("Car"), while Radar provides precise distance and speed.
3.  **Safety**: Critical for autonomous systems to have redundancy.

## Environment Setup

This project uses `conda` for environment management.

### Prerequisites
- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

### Installation

1.  **Create the environment**:
    ```bash
    conda env create -f environment.yml
    ```

2.  **Activate the environment**:
    ```bash
    conda activate radar_camera_fusion
    ```

3.  **Run the Application**:
    ```bash
    streamlit run app.py
    ```

## Directory Structure

```text
radar_camera_fusion/
├── app.py                # Main application entry point (Streamlit UI)
├── fusion_module.py      # Algorithms for associating and fusing sensor data
├── radar_module.py       # Simulates radar detections (Mock Data)
├── environment.yml       # Conda environment configuration
└── requirements.txt      # Python dependencies (pip)
```

## Evaluation Metrics

To assess the performance of the individual sensors and the fused system, the following metrics are commonly used:

### 1. Camera Object Detection Metrics
*   **mAP (mean Average Precision)**: The gold standard for object detection. It measures the average precision across all classes and IoU thresholds.
*   **IoU (Intersection over Union)**: Measures the overlap between the predicted bounding box and the ground truth box.
*   **Precision & Recall**:
    *   *Precision*: What proportion of positive identifications was actually correct?
    *   *Recall*: What proportion of actual positives was identified correctly?

### 2. Radar Detection Metrics
*   **Range Error**: The difference between the measured distance and the actual distance.
*   **Velocity Error**: The difference between measured speed and actual speed.
*   **False Alarm Rate (FAR)**: The frequency of detecting targets that do not exist (clutter/noise).
*   **Probability of Detection (Pd)**: The likelihood of detecting a target that is present.

### 3. Fusion System Metrics
*   **OSPA (Optimal Subpattern Assignment)**: A metric that captures both the cardinality error (wrong number of objects) and the state error (position/velocity accuracy) in a single score.
*   **Consistency**: How often the fused system agrees with the ground truth compared to individual sensors.
*   **ID Switches**: (If tracking is involved) How often the system loses track of an object and re-assigns it a new ID.
*   **Latency**: The time taken to process and fuse the data. Fusion should not introduce unacceptable delays.
