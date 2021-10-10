#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat img, processed;
int sigma1, sigma2;

void Gaussian(Mat &input, Mat &output, double sigma) {
    int rows = input.rows;
    int cols = input.cols;
    output = Mat::zeros(rows, cols, input.type());

    int windowSize = (int) ((6 * sigma - 1) / 2) * 2 + 1;
    int border = windowSize / 2;

    Mat input_border;
    copyMakeBorder(input, input_border, border, border, border, border, BORDER_REFLECT_101);

    vector<double> filter(windowSize, 0);
    double sum = 0;
    for (int i = 0; i < windowSize; i++) {
        int center_i = i - windowSize / 2;//[0,1,2,3,4]->[-2,-1,0,1,2]
        filter[i] = exp(-(center_i * center_i) / (2 * sigma * sigma));
        sum += filter[i];
    }
    for (auto &x:filter) x /= sum;//normalization

    for (int y = border; y < rows + border; y++) {//这里不能是input_border.rows，会越界
        for (int x = border; x < cols + border; x++) {
            double channels[3] = {0};
            for (int r = -border; r <= border; r++) {
                for (int c = 0; c < 3; c++) {
                    //水平方向
                    channels[c] += input_border.at<Vec3b>(y, x + r)[c] * filter[r + border];
                }
            }
            for (double &c : channels) {
                if (c < 0) c = 0;
                else if (c > 255) c = 255;
            }
            for (int c = 0; c < 3; c++) {
                output.at<Vec3b>(y - border, x - border)[c] = static_cast<uchar>(channels[c]);
            }
        }
    }

    copyMakeBorder(output, input_border, border, border, border, border, BORDER_REFLECT_101);
    for (int y = border; y < rows + border; y++) {
        for (int x = border; x < cols + border; x++) {
            double channels[3] = {0};
            for (int r = -border; r <= border; r++) {
                for (int c = 0; c < 3; c++) {
                    //垂直方向
                    channels[c] += input_border.at<Vec3b>(y + r, x)[c] * filter[r + border];
                }
            }
            for (double &c : channels) {
                if (c < 0) c = 0;
                else if (c > 255) c = 255;
            }
            for (int c = 0; c < 3; c++) {
                output.at<Vec3b>(y - border, x - border)[c] = static_cast<uchar>(channels[c]);
            }
        }
    }
}

void sigmaController(int, void *) {
    double sigma = 1.0 * sigma1 + 0.01 * sigma2;
    if (sigma == 0) imshow("filtering", img);
    else {
        Gaussian(img, processed, sigma);
        imshow("filtering", processed);
    }
}

int main() {
    ios::sync_with_stdio(false);

    img = imread("/home/liyang/图片/eileen.png");

    namedWindow("original img", WINDOW_AUTOSIZE);
    moveWindow("original img", 200, 200);
    imshow("original img", img);

    namedWindow("filtering", WINDOW_AUTOSIZE);
    moveWindow("filtering", 800, 100);

    createTrackbar("integer part   ", "filtering", &sigma1, 10, sigmaController);
    createTrackbar("fractional part", "filtering", &sigma2, 99, sigmaController);

    sigma1 = 0, sigma2 = 0;
    sigmaController(0, nullptr);

    waitKey(0);
    destroyAllWindows();

    return 0;
}