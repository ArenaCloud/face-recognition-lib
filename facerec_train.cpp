#include "face_rec/face_training.h"
#include <opencv2/highgui/highgui.hpp>

using namespace nl;
using namespace sping;
using namespace face_rec;

/**
 * Trains FISHERFACES, EIGENFACES and LBPH; a pre-requisite is that there must be at least 2 labels
 * This expects RAW images, faces are detected using a given cascade file.
 */
int main(int argc, const char *argv[]) {
    if (argc != 4) {
        cout << "usage: " << argv[0] << " </path/to/face_detection_cascade> </path/to/csv> </path/to/target_dir>" << endl;
        cout << "\t </path/to/face_detection_cascade> -- Path to the Haar Cascade for face detection." << endl;
        cout << "\t </path/to/csv> -- Path to the CSV file with the face database." << endl;
        cout << "\t </path/to/target_dir> -- The target path for serializing model" << endl;
        exit(1);
    }

    const string face_cascade_filename = string(argv[1]);
    const string csv_filename = string(argv[2]);
    const string target_dir = string(argv[3]);

    // TODO variables add to cli
    int target_width = 180;
    int target_height = 220;
    int min_face_width = 200;
    int min_face_height = 200;

    FaceTraining faceTraining(csv_filename, target_dir);
    // load cascade
    if (!faceTraining.get_face_recognition().load_cascade(face_cascade_filename)) {
        cout << "Could not load cascade " << face_cascade_filename << endl;
        return 1;
    }
    bool detectedAllFaces = faceTraining.detect_face_and_normalize_input(target_width, target_height, min_face_width, min_face_height);
    if (!detectedAllFaces) {
        cout << "Not all faces detected, check shown input images" << endl;
        cout << "Press any key to continue" << endl;
        waitKey(0);
    }

    cout << "Training..." << endl;
    faceTraining.train();

    cout << "Saving to " << target_dir << endl;
    faceTraining.save();

    return 0;
}