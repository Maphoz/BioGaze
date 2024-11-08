# BioGaze: Face Quality Analysis for ICAO/ISO Standards

## Description
*// Placeholder for general description of the BioGaze software.*

---

### Core Tasks

BioGaze performs face quality analysis based on three essential tasks, each contributing to ICAO and ISO standard compliance:

1. **Face Detection** - Locates and isolates faces within images.
2. **Landmark Detection** - Identifies key facial features, such as the eyes, nose, and mouth.
3. **Face Parsing** - Segments facial regions to analyze structural qualities of the face.

<div align="center">
  <img src="resources/test.png" alt="Base Image" width="200" />
  <img src="resources/test_detect.png" alt="Face detection" width="200" />
  <img src="resources/test_landmark.png" alt="Landmark detection" width="200" />
  <img src="resources/test_parsing.png" alt="Face parsing" width="200" />
</div>

---

### Additional AI Models

To further enhance analysis, BioGaze integrates several AI models:

1. **Gaze Estimation** - Determines if the gaze is directed towards the camera.
2. **Emotion Recognition** - Identifies the emotional expression, ensuring compliance with a neutral expression.
3. **Head Pose Estimation** - Assesses the orientation of the face for frontal pose requirements.

<div align="center">
  <img src="resources/emo_test.png" alt="Base Image" width="200" />
  <img src="resources/emo_rec.png" alt="Emotion recognition" width="200" />
</div>
<div align="center">
  <img src="resources/gaze_img.png" alt="Base Image" width="200" />
  <img src="resources/gaze.png" alt="Gaze estimation" width="200" />
</div>
