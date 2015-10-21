#ifndef OPENCVFACERECTEST_FACE_TRAINING_H
#define OPENCVFACERECTEST_FACE_TRAINING_H

#include <string>
#include "opencv2/contrib/contrib.hpp"
#include "face_recognition.h"

using namespace cv;
using namespace std;

namespace nl {
    namespace sping {
        namespace face_rec {

            class FaceTraining {

            public:
                /**
                 * filename is the csv file containing the paths and labels of the images.
                 * target_dir is the directory where the models will be serialized
                 */
                FaceTraining(const string &csv_filename, const string &target_dir);

                /**
                 * target_dir is the directory where the models will be loaded from
                 */
                FaceTraining(const string &target_dir);

                virtual ~FaceTraining() { }

                static void read_csv(const string &csv_filename, vector<Mat> &images, vector<int> &labels,
                                     char separator = ';');

                FaceRecognition &get_face_recognition();

                /**
                 * Transforms the original_images into a specific size target_width and target_height
                 */
                void normalize_input(int target_width, int target_height);

                /**
                 * Transforms the original_images into a specific size target_width and target_height
                 * and detects faces with a min_face_width and min_face_height
                 */
                void detect_face_and_normalize_input(int target_width, int target_height, int min_face_width, int min_face_height);

                /**
                 * Trains the (image, label) pairs given in the CSV
                 */
                void train();

                /**
                 * Serializes the models to target dir
                 */
                void save();

                /*
                 * Deserializes the models from target dir
                 */
                void load();

                vector<Mat> &get_original_images();

            private:
                string target_dir;
                vector<Mat> original_images;
                vector<int> original_labels;
                vector<Mat> normalized_images;
                vector<int> normalized_labels;
                FaceRecognition face_recognition;
            };
        }
    }
}

#endif //OPENCVFACERECTEST_FACE_TRAINING_H
