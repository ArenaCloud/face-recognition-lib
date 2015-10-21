#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "face_rec/face_recognition.h"
#include "face_rec/face_training.h"

#include <unistd.h>

using namespace cv;
using namespace std;
using namespace nl;
using namespace sping;
using namespace face_rec;

static void normalize_input(Mat &src, int src_label,
                            vector<Mat> &faces_resized, vector<int> &labels_resized,
                            int target_width, int target_height, CascadeClassifier classifier,
                            int min_width_face = -1, int min_height_face = -1
) {
    FaceRecognition::detect_face_and_normalize_input(src, src_label, faces_resized, labels_resized, target_width,
                                                     target_height,
                                                     classifier, min_width_face, min_height_face);

}

static void normalize_input_webcam(
        VideoCapture cap,
        vector<Mat> &faces_resized, vector<int> &labels_resized,
        int width, int height,
        CascadeClassifier classifier, int face_width, int face_height,
        int label_offset) {
    Mat frame;
    bool captureNext = false;
    int current_label = label_offset;
    size_t start = faces_resized.size();
    for(;;) {
        cap >> frame;
        Mat original = frame.clone();
        // Convert frame to grayscale
        Mat gray;
        cvtColor(original, gray, CV_BGR2GRAY);
        vector<Rect> faces;
        classifier.detectMultiScale(gray, faces, 1.1, 3, 0, Size(face_width, face_height));
        // largest face
        int largest_area = -1;
        Rect largest_rect;
        for(size_t i = 0; i < faces.size(); i++) {
            Rect face_i = faces[i];
            if (largest_area == -1) {
                largest_area = face_i.area();
                largest_rect = face_i;
            } else {
                if (largest_area < face_i.area()) {
                    largest_area = face_i.area();
                    largest_rect = face_i;
                }
            }
        }

        if (captureNext && faces.size() > 0) {
            Mat face_resized;
            cv::resize(gray(largest_rect), face_resized, Size(width, height), 1.0, 1.0, INTER_CUBIC);
            faces_resized.push_back(face_resized);
            labels_resized.push_back(current_label);
            captureNext = false;
        }

        if (start < faces_resized.size()) {
            imshow("last_captured", faces_resized[faces_resized.size() - 1]);
        }

        imshow("training stream", original);

        char key = (char) waitKey(20);

        if (key == 49) { // 1
            break;
        }

        if (key == 50) { // 2
            captureNext = true;
        }

        if (key == 51) {
            current_label += 1;
        }
    }
    destroyAllWindows();

}

/**
 * Trains a given set of (image,label) pairs, optionally training with captured webcam images.
 *
 * This consists of a few steps:
 * 1. Capture several frames using the key "2", the extracted face is then shown in a window.
 * 2. Press "1" to finish, press "3" to increment the label goto 1.
 * After this the preview images are shown of the (extracted) faces to be trained on.
 * 3. Press ESC to end the preview
 * Now the models will be trained, after training is finished a new window is opened which captures frames from the webcam.
 * 4. Use the keys 1, 2, 3 to switch between respectively FISHERFACES, EIGENFACES and LBPH
 * Use ESC to exit
 *
 */
int main(int argc, const char *argv[]) {
    if (argc != 4) {
        cout << "usage: " << argv[0] << " </path/to/haar_cascade> </path/to/csv.ext> </path/to/device id>" << endl;
        cout << "\t </path/to/haar_cascade> -- Path to the Haar Cascade for face detection." << endl;
        cout << "\t </path/to/csv.ext> -- Path to the CSV file with the face database." << endl;
        cout << "\t <device id> -- The webcam device id to grab frames from." << endl;
        exit(1);
    }
    string fn_haar = string(argv[1]);
    string fn_csv = string(argv[2]);
    int deviceId = atoi(argv[3]);

    vector<Mat> images;
    vector<int> labels;
    try {
        FaceTraining::read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }

    // target face scale size
    int im_width = 180;
    int im_height = 220;

    // face detection in prediction
    int pred_width = -1;
    int pred_height = -1;

    int min_face_width = 300;
    int min_face_height = 300;

    vector<Mat> faces_resized;
    vector<int> faces_resized_labels;

    // Get a handle to the Video device:
    VideoCapture cap(deviceId);
    // Check if we can use this device at all:
    if(!cap.isOpened()) {
        cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
        return -1;
    }

    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1600);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1200);

    FaceRecognition face_rec;
    face_rec.load_cascade(fn_haar);

    for (size_t i = 0; i < images.size(); i++) {
        imshow(format("%d", i), images[i]);
    }

    for (size_t i = 0; i < images.size(); i++) {
        normalize_input(
                images[i], labels[i],
                faces_resized, faces_resized_labels,
                im_width, im_height, face_rec.face_detection_classifier(),
                min_face_width, min_face_height
        );
    }

    int label_offset = -1;
    for (size_t i = 0; i < faces_resized_labels.size(); i++) {
        if (label_offset <= faces_resized_labels[i]) {
            label_offset = faces_resized_labels[i] + 1;
        }
    }

    // train with webcam
    normalize_input_webcam(cap, faces_resized, faces_resized_labels,
                           im_width, im_height, face_rec.face_detection_classifier(),
                           min_face_width, min_face_height,
                           label_offset
    );


    // preview detected faces
    int c = 0;
    Mat face;
    int label;
    for (;;) {
        face = faces_resized[c % faces_resized.size()];
        label = faces_resized_labels[c % faces_resized.size()];
        usleep(100000);
        imshow(format("label %d w: %d h: %d", label, face.cols, face.rows), face);
        char key = (char) waitKey(20);
        c++;
        if(key == 27)
            break;
    }
    destroyAllWindows();


    // Create a FaceRecognizer and train it on the given images:
    face_rec.train(FISHERFACES, faces_resized, faces_resized_labels);
    face_rec.train(EIGENFACES, faces_resized, faces_resized_labels);
    face_rec.train(LBPH, faces_resized, faces_resized_labels);
    OpenCVFaceRecognizer current_recognizer = LBPH;

    // Holds the current frame from the Video device:
    Mat frame;
    for(;;) {
        cap >> frame;
        // Clone the current mesme:
        Mat original = frame.clone();
        int pred = -1000;
        double confidence = 0.0;

        face_rec.predict(current_recognizer, original, pred, confidence, im_width, im_height, pred_width, pred_height);

        imshow("face_recognizer", original);
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

