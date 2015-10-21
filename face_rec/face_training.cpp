#include "face_training.h"
#include <fstream>
#include <opencv2/highgui/highgui.hpp>

namespace nl {
    namespace sping {
        namespace face_rec {

            const string FISHERFACES_FILENAME = "fisherfaces_model";
            const string EIGENFACES_FILENAME = "eigenfaces_model";
            const string LBPH_FILENAME = "lbph_model";

            FaceTraining::FaceTraining(const string &csv_filename, const string &target_dir) {
                read_csv(csv_filename, this->original_images, this->original_labels);
                this->target_dir = target_dir;
            }

            FaceTraining::FaceTraining(const string &target_dir) {
                this->target_dir = target_dir;
            }

            void FaceTraining::detect_face_and_normalize_input(int target_width, int target_height, int min_face_width, int min_face_height) {
                CascadeClassifier classifier = this->get_face_recognition().face_detection_classifier();
                for (size_t i = 0; i < this->original_images.size(); i++) {
                    FaceRecognition::detect_face_and_normalize_input(
                            this->original_images[i], this->original_labels[i],
                            this->normalized_images, this->normalized_labels,
                            target_width, target_height, classifier,
                            min_face_width, min_face_height
                    );
                }
            }

            void FaceTraining::normalize_input(int target_width, int target_height) {
                for (size_t i = 0; i < this->original_images.size(); i++) {
                    FaceRecognition::normalize_input(
                            this->original_images[i], this->original_labels[i],
                            this->normalized_images, this->normalized_labels,
                            target_width, target_height
                    );
                }
            }

            void FaceTraining::train() {
                this->face_recognition.train(FISHERFACES, this->normalized_images, this->normalized_labels);
                this->face_recognition.train(EIGENFACES, this->normalized_images, this->normalized_labels);
                this->face_recognition.train(LBPH, this->normalized_images, this->normalized_labels);
            }

            void FaceTraining::save() {
                this->face_recognition.save(FISHERFACES, this->target_dir + "/" + FISHERFACES_FILENAME);
                this->face_recognition.save(EIGENFACES, this->target_dir + "/" + EIGENFACES_FILENAME);
                this->face_recognition.save(LBPH, this->target_dir + "/" + LBPH_FILENAME);
            }

            void FaceTraining::load() {
                this->face_recognition.load(FISHERFACES, this->target_dir + "/" + FISHERFACES_FILENAME);
                this->face_recognition.load(EIGENFACES, this->target_dir + "/" + EIGENFACES_FILENAME);
                this->face_recognition.load(LBPH, this->target_dir + "/" + LBPH_FILENAME);
            }

            FaceRecognition &FaceTraining::get_face_recognition() {
                return this->face_recognition;
            }


            void FaceTraining::read_csv(const string &csv_filename, vector<Mat> &images,
                                        vector<int> &labels,
                                        char separator) {
                std::ifstream file(csv_filename.c_str(), ifstream::in);
                if (!file) {
                    string error_message = "No valid input file was given, please check the given filename.";
                    CV_Error(CV_StsBadArg, error_message);
                }
                string line, path, classlabel;
                while (getline(file, line)) {
                    stringstream liness(line);
                    getline(liness, path, separator);
                    getline(liness, classlabel);
                    if (!path.empty() && !classlabel.empty()) {
                        images.push_back(imread(path, CV_LOAD_IMAGE_GRAYSCALE));
                        labels.push_back(atoi(classlabel.c_str()));
                    }
                }
            }

            vector<Mat> &FaceTraining::get_original_images() {
                return original_images;
            }
        }
    }
}