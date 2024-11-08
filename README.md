# BioGaze: Face Quality Analysis for ICAO/ISO Standards

## Description
*// Placeholder for general description of the BioGaze software.*

---

## Requirements Verified

<div align="center">

<table>
  <thead>
    <tr>
      <th style="background-color: black; color: white; padding: 8;">Subject</th>
      <th style="background-color: black; color: white; padding: 8;">Photographic</th>
      <th style="background-color: black; color: white; padding: 8;">Acquisition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Head without covering</td>
      <td>Correct exposure</td>
      <td>Uniform background</td>
    </tr>
    <tr>
      <td>Gaze in camera</td>
      <td>In focus photo</td>
      <td>Uniform face lighting</td>
    </tr>
    <tr>
      <td>No/light makeup</td>
      <td>Correct saturation</td>
      <td>No pixelation</td>
    </tr>
    <tr>
      <td>Neutral expression</td>
      <td>Proper face dimension</td>
      <td>No posterization</td>
    </tr>
    <tr>
      <td>No sunglasses</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Eyes open</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Frontal Pose</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>

</div>


### Core Tasks

BioGaze performs face quality analysis based on three essential tasks, each contributing to ICAO and ISO standard compliance:

1. **Face Detection** - Locates and isolates faces within images.
2. **Landmark Detection** - Identifies key facial features, such as the eyes, nose, and mouth.
3. **Face Parsing** - Segments facial regions to analyze structural qualities of the face.

<div align="center">
  <img src="resources_readme/test.png" alt="Base Image" width="200" />
  <img src="resources_readme/test_detect.png" alt="Face detection" width="200" />
  <img src="resources_readme/test_landmark.png" alt="Landmark detection" width="200" />
  <img src="resources_readme/test_parsing_2.png" alt="Face parsing" width="200" />
</div>

<p align="center">
  Base Image &nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp; Face Detection &nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp; Landmark Detection &nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp; Face Parsing
</p>



---

### Additional AI Models

To further enhance analysis, BioGaze integrates several AI models:

1. **Gaze Estimation** - Determines if the gaze is directed towards the camera.
2. **Emotion Recognition** - Identifies the emotional expression, ensuring compliance with a neutral expression.
3. **Head Pose Estimation** - Assesses the orientation of the face for frontal pose requirements.

<div align="center">
  <img src="resources_readme/emo_test.png" alt="Base Image" width="250" />
  <img src="resources_readme/emo_rec.png" alt="Emotion recognition" width="250" />
</div>
<div align="center">
  <img src="resources_readme/gaze_img.png" alt="Base Image" width="250" />
  <img src="resources_readme/gaze.png" alt="Gaze estimation" width="250" />
</div>

---

## How to Install

1. **Clone the Repository**

First, clone the BioGaze repository from GitHub:

```bash
  git clone https://github.com/Maphoz/BioGaze.git
  cd BioGaze

2. **Install Dependencies**

Install the required Python packages as specified in the requirements.txt file:

```bash
  pip install -r requirements.txt
