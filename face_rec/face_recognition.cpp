#include "face_recognition.h"

namespace nl {
    namespace sping {
        namespace face_rec {

            const map<OpenCVFaceRecognizer, string> FaceRecognition::OpenCVFaceRecognizerToString = {
                    {FISHERFACES, "FISHERFACES"},
                    {EIGENFACES, "EIGENFACES"},
                    {LBPH, "LBPH"}
            };

            const map<string, OpenCVFaceRecognizer> FaceRecognition::OpenCVFaceRecognizerStringToEnum = {
                    {"FISHERFACES", FISHERFACES},
                    {"EIGENFACES", EIGENFACES},
                    {"LBPH", LBPH}
            };

            void FaceRecognition::set_recognizer(OpenCVFaceRecognizer recognizer,
                                                 Ptr<FaceRecognizer> &open_cv_recognizer) {
                _recognizers[recognizer] = open_cv_recognizer;
            }

            void FaceRecognition::train(OpenCVFaceRecognizer recognizer, const _InputArray &src,
                                        const _InputArray &labels) {
                get_recognizer(recognizer)->train(src, labels);
            }

            bool FaceRecognition::has_recognizer(OpenCVFaceRecognizer recognizer) {
                map<OpenCVFaceRecognizer, Ptr<FaceRecognizer> >::iterator it = _recognizers.find(recognizer);
                return _recognizers.end() != it;
            }

            Ptr<FaceRecognizer> FaceRecognition::create_recognizer(OpenCVFaceRecognizer recognizer) {
                Ptr<FaceRecognizer> open_cv_recognizer;

                switch (recognizer) {
                    case FISHERFACES:
                        open_cv_recognizer = createFisherFaceRecognizer();
                        break;
                    case EIGENFACES:
                        open_cv_recognizer = createEigenFaceRecognizer();
                        break;
                    case LBPH:
                        open_cv_recognizer = createLBPHFaceRecognizer();
                        break;
                }

                return open_cv_recognizer;
            }

            Ptr<FaceRecognizer> FaceRecognition::get_recognizer(OpenCVFaceRecognizer recognizer) {
                Ptr<FaceRecognizer> open_cv_recognizer;
                if (!has_recognizer(recognizer)) {
                    open_cv_recognizer = create_recognizer(recognizer);
                    set_recognizer(recognizer, open_cv_recognizer);
                    return get_recognizer(recognizer);
                }
                return _recognizers[recognizer];
            }

            void FaceRecognition::save(OpenCVFaceRecognizer recognizer, const string &file_name) {
                get_recognizer(recognizer)->save(file_name);
            }

            void FaceRecognition::load(OpenCVFaceRecognizer recognizer, const string &file_name) {
                get_recognizer(recognizer)->load(file_name);
            }

            bool FaceRecognition::load_cascade(const string &file_name) {
                return _face_detection_classifier.load(file_name);
            }

            unsigned long FaceRecognition::detect_face_and_normalize_input(Mat &src, int src_label, vector<Mat> &dst,
                                                                  vector<int> &dst_labels, int target_width,
                                                                  int target_height, CascadeClassifier &cascade,
                                                                  int min_width_face, int min_height_face
            ) {
                // Find the faces in the frame:
                vector<Rect> rects;
                cascade.detectMultiScale(src, rects, 1.1, 3, 0, Size(min_width_face, min_height_face));

                for (Rect& rect : rects) {
                    Mat face_resized;

                    // check minimum rectangle size
                    cv::resize(src(rect), face_resized, Size(target_width, target_height), 1.0, 1.0, INTER_CUBIC);
                    dst.push_back(face_resized);
                    dst_labels.push_back(src_label);
                }
                return rects.size();
            }

            void FaceRecognition::normalize_input(Mat &src, int src_label, vector<Mat> &dst,
                                                  vector<int> &dst_label, int target_width,
                                                  int target_height) {
                Mat face_resized;

                cv::resize(src.clone(), face_resized, Size(target_width, target_height), 1.0, 1.0, INTER_CUBIC);
                dst.push_back(face_resized);
                dst_label.push_back(src_label);
            }

            void FaceRecognition::predict(OpenCVFaceRecognizer recognizer, Mat &face_gray_resized, int& label, double& confidence) {
                get_recognizer(recognizer)->predict(face_gray_resized, label, confidence);
            }

            void FaceRecognition::predict(vector<OpenCVFaceRecognizer> &recognizers, Mat &src, vector<int> &labels,
                                          vector<double> &confidences, int target_width, int target_height,
                                          int min_face_width, int min_face_height) {
                assert(recognizers.size() == labels.size() && labels.size() == confidences.size());

                Mat src_gray;
                cvtColor(src, src_gray, CV_BGR2GRAY);
                vector<Rect> face_rectangles;
                _face_detection_classifier.detectMultiScale(src_gray, face_rectangles, 1.1, 3, 0, Size(min_face_width, min_face_height));

                // if no faces are detected, return
                if (face_rectangles.size() == 0) {
                    return;
                }

                // get largest face
                int largest_area = -1;
                Rect largest_rect;
                for(auto &face_rect : face_rectangles) {
                    if (face_rect.area() > largest_area) {
                        largest_rect = face_rect;
                        largest_area = face_rect.area();
                    }
                }
                Mat face_resized;
                Mat cropped_face = src_gray(largest_rect);
                cv::resize(cropped_face, face_resized, Size(target_width, target_height), 1.0, 1.0, INTER_CUBIC);

                for (size_t i = 0; i < recognizers.size(); i++) {
                    predict(recognizers[i], face_resized, labels[i], confidences[i]);
                }

                rectangle(src, largest_rect, CV_RGB(0, 255,0), 1);
                int pos_x = std::max(largest_rect.tl().x - 10, 0);
                int pos_y = std::max(largest_rect.tl().y - 10, 0);
                for (size_t i = 0; i < recognizers.size(); i++) {
                    string algorithm = FaceRecognition::OpenCVFaceRecognizerToString.at(recognizers[i]);
                    string box_text = format("Alg: %s Label: %d, Confidence: %.2f", algorithm.c_str(), labels[i], confidences[i]);
                    putText(src, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2);
                    pos_y -= 15;
                }


            }

            void FaceRecognition::predict(OpenCVFaceRecognizer recognizer, Mat &src, int& label, double &confidence,
                                          int target_width, int target_height,
                                          int min_face_width, int min_face_height) {
                vector<OpenCVFaceRecognizer> recognizers = {recognizer};
                vector<int> labels = {label};
                vector<double> confidences = {confidence};
                predict(recognizers, src, labels, confidences, target_width, target_height, min_face_width, min_face_height);
            }

            CascadeClassifier &FaceRecognition::face_detection_classifier() {
                return _face_detection_classifier;
            }
        }
    }
}
