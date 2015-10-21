#include "face_recognition.h"

namespace nl {
    namespace sping {
        namespace face_rec {

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

            void FaceRecognition::load_cascade(const string &file_name) {
                _face_detection_classifier.load(file_name);
            }

            void FaceRecognition::detect_face_and_normalize_input(Mat &src, int src_label, vector<Mat> &dst,
                                                                  vector<int> &dst_labels, int target_width,
                                                                  int target_height, CascadeClassifier &cascade,
                                                                  int min_width_face, int min_height_face
            ) {
                // Find the faces in the frame:
                vector<Rect> rects;
                cascade.detectMultiScale(src, rects, 1.1, 3, 0, Size(min_width_face, min_height_face));

                Rect rect;
                for(size_t j = 0; j < rects.size(); j++) {
                    rect = rects[j];
                    Mat face_resized;

                    // check minimum rectangle size
                    cv::resize(src(rect), face_resized, Size(target_width, target_height), 1.0, 1.0, INTER_CUBIC);
                    dst.push_back(face_resized);
                    dst_labels.push_back(src_label);
                }
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

            void FaceRecognition::predict(OpenCVFaceRecognizer recognizer, Mat &src, int& label, double &confidence,
                                          int target_width, int target_height,
                                          int min_face_width, int min_face_height) {
                Mat src_gray;
                cvtColor(src, src_gray, CV_BGR2GRAY);
                vector<Rect> face_rectangles;
                _face_detection_classifier.detectMultiScale(src_gray, face_rectangles, 1.1, 3, 0, Size(min_face_width, min_face_height));
                int largestRect = -1;
                for(size_t i = 0; i < face_rectangles.size(); i++) {
                    // Process face by face:
                    Rect face_i = face_rectangles[i];
                    Mat cropped_face = src_gray(face_i);
                    Mat face_resized;
                    cv::resize(cropped_face, face_resized, Size(target_width, target_height), 1.0, 1.0, INTER_CUBIC);

                    int tmp_pred = -1;
                    double tmp_confidence = 0.0;
                    predict(recognizer, face_resized, tmp_pred, tmp_confidence);

                    // gets largest face
                    if (face_i.area() > largestRect) {
                        largestRect = face_i.area();
                        label = tmp_pred;
                        confidence = tmp_confidence;
                    }

                    rectangle(src, face_i, CV_RGB(0, 255,0), 1);
                    // Create the text we will annotate the box with:
                    string box_text = format("Prediction = %d %.2f", tmp_pred, tmp_confidence);
                    int pos_x = std::max(face_i.tl().x - 10, 0);
                    int pos_y = std::max(face_i.tl().y - 10, 0);

                    putText(src, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2);
                }
            }

            CascadeClassifier &FaceRecognition::face_detection_classifier() {
                return _face_detection_classifier;
            }
        }
    }
}
