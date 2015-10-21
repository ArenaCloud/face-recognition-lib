#ifndef OPENCVFACERECTEST_FACERECOGNITION_H
#define OPENCVFACERECTEST_FACERECOGNITION_H

#include <string>
#include <iostream>
#include "opencv2/contrib/contrib.hpp"

using namespace std;
using namespace cv;

namespace nl {
    namespace sping {
        namespace face_rec {

            enum OpenCVFaceRecognizer {FISHERFACES, EIGENFACES, LBPH};

            class FaceRecognition {

                map<OpenCVFaceRecognizer, Ptr<FaceRecognizer> > _recognizers;
                CascadeClassifier _face_detection_classifier;

                /**
                 * Convenience method to check whether a certain recognizer is initialized
                 */
                bool has_recognizer(OpenCVFaceRecognizer recognizer);

                /**
                 * Creates a default Ptr<FaceRecognizer> by type
                 */
                Ptr<FaceRecognizer> create_recognizer(OpenCVFaceRecognizer recognizer);

            public:
                FaceRecognition() { }

                /**
                 * Resizes src and puts the resized image in the dst vector.
                 */
                static void normalize_input(Mat &src, int src_label,
                                            vector<Mat> &dst, vector<int> &dst_label, int target_width,
                                            int target_height);

                /**
                 * Detects a face/faces in src, extracts and resizes face and puts it in the dst vector
                 */
                static void detect_face_and_normalize_input(Mat &src, int src_label,
                                                            vector<Mat> &dst, vector<int> &dst_labels,
                                                            int target_width, int target_height,
                                                            CascadeClassifier &cascade,
                                                            int min_width_face = -1, int min_height_face = -1);

                /**
                 * Returns the CascadeClassifier for face_detection
                 */
                CascadeClassifier &face_detection_classifier();

                /**
                 * Initializes CascadeClassifier with the given file
                 */
                void load_cascade(const string& file_name);

                /**
                 * Trains the recognizer with given images and labels.
                 */
                void train(OpenCVFaceRecognizer recognizer, InputArrayOfArrays src, InputArray labels);

                /**
                 * Puts a certain Ptr<FaceRecognizer> in the _recognizers map
                 * Using this method it's possible initialize a FaceRecognizer with custom parameters
                 */
                void set_recognizer(OpenCVFaceRecognizer recognizer, Ptr<FaceRecognizer>& open_cv_recognizer);

                /**
                 * Retrieves a specified Ptr<FaceRecognizer>, if the recognizer does not exist, then it will be
                 * constructed using default parameters.
                 */
                Ptr<FaceRecognizer> get_recognizer(OpenCVFaceRecognizer recognizer);

                /**
                 * Loads a certain recognizer with the serialized model
                 */
                void load(OpenCVFaceRecognizer recognizer, const string& file_name);

                /**
                 * Saves a certain recognizer, serializing it to the given file
                 */
                void save(OpenCVFaceRecognizer recognizer, const string& file_name);

                /**
                 * Detects faces in src, returns label and confidence of largest detected face
                 * Draws rectangles over detected faces with label and confidence
                 *
                 * Returns label and confidence given a face and a recognizer
                 * Note that the input image must be of the correct dimensions if using FISHERFACES or EIGENFACES
                 */
                void predict(OpenCVFaceRecognizer recognizer, Mat& src, int& label, double& confidence,
                             int target_width, int target_height,
                             int min_face_width, int min_face_height);

                /**
                 * Returns label and confidence given a face and a recognizer
                 * Note that the input image must be of the correct dimensions if using FISHERFACES or EIGENFACES
                 */
                void predict(OpenCVFaceRecognizer recognizer, Mat &face_gray_resized, int &label, double &confidence);


                virtual ~FaceRecognition() {}
            };



        }
    }
}



#endif //OPENCVFACERECTEST_FACERECOGNITION_H
