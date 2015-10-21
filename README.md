# face-recognition-lib

## Prerequisites

 * OpenCV 2.x
 * Some C++11 compiler
 * CMake 3.x
 
## Overview
This repository consists of a face_rec library and three example programs which use the library. 

### The library
There are two classes FaceRecognition and FaceTraining.

 * FaceTraining is used to train models and serialize/deserialize trained models.
 * FaceRecognition is wrapper of OpenCV's FaceRecognizers. 
 
### Example programs

 * facerec_detect - Predicts faces from a webcam given models and cascade file
 * facerec_train - Trains faces given CSV and serializes the models
 * facerec_video - Trains faces given CSV and can train faces from webcam.
 
### CSV format

The line format of the CSV file is:
```
<absolute_path>;<label>
```
where **absolute_path** is an absolute path on your local file system and **label** is an **integer**.

### Disclaimer

Basis of this library lies in the FaceRecognizer introduction at http://docs.opencv.org/modules/contrib/doc/facerec/facerec_tutorial.html