#include <opencv2/highgui/highgui.hpp>
#include "face_rec/face_training.h"

using namespace nl;
using namespace sping;
using namespace face_rec;

/**
 * Draws a rectangle over the detected face, together with the label and confidence interval.
 * Use the keys 1, 2, 3 to switch between respectively  FISHERFACES, EIGENFACES and LBPH
 * Use ESC to exit
 */
int main(int argc, const char *argv[]) {

    if (argc != 4) {
        cout << "usage: " << argv[0] << " </path/to/face_detection_cascade> </path/to/target_dir> <device id>" << endl;
        cout << "\t </path/to/face_detection_cascade> -- Path to the Haar Cascade for face detection." << endl;
        cout << "\t </path/to/target_dir> -- Path of the trained models" << endl;
        cout << "\t <device id> -- The webcam device id to grab frames from." << endl;
        exit(1);
    }

    string face_cascade_filename = string(argv[1]);
    string target_dir = string(argv[2]);
    int deviceId = atoi(argv[3]);

    FaceTraining faceTraining(target_dir);
    faceTraining.get_face_recognition().load_cascade(face_cascade_filename);
    faceTraining.load();

    int target_width = 180;
    int target_height = 220;
    int min_face_width = -1;
    int min_face_height = -1;

    VideoCapture cap(deviceId);
    if(!cap.isOpened()) {
        cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
        return -1;
    }

//    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1600);
//    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1200);

    OpenCVFaceRecognizer current_recognizer = LBPH;

    Mat frame;
    for(;;) {
        cap >> frame;
        Mat original = frame.clone();
        int pred = -1000;
        double confidence = 0.0;

        faceTraining.get_face_recognition().predict(
                current_recognizer, original,
                pred, confidence, target_width, target_height, min_face_width, min_face_height);

        // Show the result:
        imshow("Face recognizer", original);
        // And display it:
        char key = (char) waitKey(20);
        bool exit = false;
        // Exit this loop on escape:
        switch(key) {
            case 27: // ESC
                exit = true;
                break;
            case 49: // 1
                current_recognizer = FISHERFACES;
                break;
            case 50: // 2
                current_recognizer = EIGENFACES;
                break;
            case 51: // 3
                current_recognizer = LBPH;
                break;
            default:
                break;
        }
        if(exit)
            break;
    }


    return 0;
}
