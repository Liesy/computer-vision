#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const char *img_path = "/home/liyang/study/cv/Circle Detection Dataset/39.jpg";

Mat img;
int min_dist, min_r, max_r, cannyThreshold1, cannyThreshold2, accThreshold;

bool cmp(Point &a, Point &b) {//点的位置从大到小排序
    if (a.y != b.y) return a.y > b.y;
    return a.x > b.x;
}

void showImage(Mat &image, const char *windowName, int location_x, int location_y) {
    namedWindow(windowName, WINDOW_AUTOSIZE);
    moveWindow(windowName, location_x, location_y);
    imshow(windowName, image);
}

void circleDetection(Mat &src, int minDist, int minR, int maxR, int canny_threshold1, int canny_threshold2,
                     int acc_threshold) {
    if (src.empty()) CV_Error(Error::StsNullPtr, "no src img");
    int rows = src.rows, cols = src.cols;

    Mat processed;
    src.copyTo(processed);

    Mat src_gray;//转换为灰度图
    cvtColor(src, src_gray, COLOR_RGB2GRAY);
    GaussianBlur(src_gray, src_gray, Size(9, 9), 2, 2);

    Mat dx, dy;//水平梯度和垂直梯度
    Sobel(src_gray, dx, CV_32F, 1, 0);
    Sobel(src_gray, dy, CV_32F, 0, 1);

    Mat src_edges;//得到边缘图像
    Canny(src_gray, src_edges, canny_threshold1, canny_threshold2);

    showImage(src_edges, "edges", 600, 200);

    Mat accum = Mat::zeros(rows, cols, CV_32SC1);//累加器矩阵
    vector<Point> nz, centers;//圆周序列和圆心序列

    //对边缘图像遍历，计算累加和
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            //如果当前的像素不是边缘点，或者水平梯度值和垂直梯度值都为0，则一定不是圆周上的点
            if (!src_edges.at<uchar>(y, x) || (dx.at<int>(y, x) == 0 && dy.at<int>(y, x) == 0)) continue;

            float current_dx = dx.at<float>(y, x), current_dy = dy.at<float>(y, x);;//当前点的梯度值
            float tanX = current_dy / current_dx;//tan(梯度方向与x轴的角度)

            for (int r = minR; r < maxR; r++) {
                float theta = -0.1;
                while (theta <= 0.1) {
                    //水平和垂直方向的位移量（即梯度方向）
                    int step_x = r * cvRound(cos(atan(tanX)) + theta);
                    int step_y = r * cvRound(sin(atan(tanX)) + theta);

                    //在当前点沿着梯度方向对经过的像素进行累加
                    int x1 = x + step_x, y1 = y + step_y;
                    if (0 <= x1 && x1 < cols && 0 <= y1 && y1 < rows) accum.at<int>(y1, x1)++;

                    theta += 0.05;
                }
            }
            //圆周上点的坐标压入圆周序列nz中
            nz.emplace_back(Point(x, y));
        }
    }
    if (nz.empty()) return;
    cout << "point num in circle:" << nz.size() << '\n';

    //遍历累加矩阵，找到可能的圆心
    for (int y = 1; y < rows - 1; y++) {
        for (int x = 1; x < cols - 1; x++) {
            int top = accum.at<int>(y - 1, x);
            int left = accum.at<int>(y, x - 1);
            int center = accum.at<int>(y, x);
            int right = accum.at<int>(y, x + 1);
            int bottom = accum.at<int>(y + 1, x);
            //当前的值大于阈值，并在4邻域内它是最大值，则该点被认为是圆心
            if (center > acc_threshold && center > top && center > left && center > right && center > bottom)
                centers.emplace_back(Point(x, y));
        }
    }
    if (centers.empty()) return;
    cout << "candidate center num:" << centers.size() << '\n';
    sort(centers.begin(), centers.end(), cmp);//对候选圆心按照由大到小的顺序进行排序

    vector<Point> circles;//最终确定的圆心
    for (auto &p:centers) {
        //使用minDist进行过滤
        int i;
        for (i = 0; i < circles.size(); i++) {
            if (pow(circles[i].x - p.x, 2) + pow(circles[i].y - p.y, 2) < minDist * minDist) break;
        }
        if (i < circles.size()) continue;

        //得到候选半径值(保存半径的平方)，这里要注意:没有去重，为了计数
        vector<int> candidate_r;
        for (auto &p_nz:nz) {//遍历圆周上的点
            double r2 = pow(p_nz.x - p.x, 2) + pow(p_nz.y - p.y, 2);
            if (minR * minR < r2 && r2 < maxR * maxR) candidate_r.emplace_back(cvRound(sqrt(r2)));
        }
        if (candidate_r.empty()) continue;

        //确定最终半径
        sort(candidate_r.begin(), candidate_r.end());
        int start_idx = 0, max_count = 0;
        int r_best, r_pre = candidate_r[0];
        for (int idx = 1; idx < candidate_r.size(); idx++) {//NOLINT
            int r = candidate_r[idx];
            if (r > maxR) break;
            if (r - r_pre < 1) continue;

            //idx-start_idx表示当前得到的相同半径的数量
            //(idx+start_idx)/2表示idx和start_idx中间的数
            //取中间的数所对应的半径值作为当前半径值r_cur，也就是取那些半径值相同的值
            int r_cur = candidate_r[(idx + start_idx) / 2];
            if (idx - start_idx > max_count) {
                r_best = r_cur;
                max_count = idx - start_idx;
            }
            r_pre = r;
            start_idx = idx;
        }
        circles.emplace_back(p);
        //画圆心
        circle(processed, p, 3, Scalar(255, 0, 0));
        //画圆
        circle(processed, p, cvRound(r_best), Scalar(0, 0, 255));
    }
    cout << "circle num:" << circles.size() << '\n';

    showImage(processed, "detection", 800, 200);
}

void controller(int, void *) {
    circleDetection(img, min_dist, min_r, max_r, cannyThreshold1, cannyThreshold2, accThreshold);
}

int main() {
    ios::sync_with_stdio(false);

    img = imread(img_path);

    string win1 = "origin";
    showImage(img, "origin", 200, 200);

    /*
    //18.jpg
    cannyThreshold1 = 115, cannyThreshold2 = 300;
    accThreshold = 25;
    min_r = 15, max_r = 40;
    min_dist = 20;
    circleDetection(img, min_dist, min_r, max_r, cannyThreshold1, cannyThreshold2, accThreshold);
    */

    //39.jpg
    cannyThreshold1 = 115, cannyThreshold2 = 300;
    accThreshold = 25;
    min_r = 15, max_r = 40;
    min_dist = 20;
    circleDetection(img, min_dist, min_r, max_r, cannyThreshold1, cannyThreshold2, accThreshold);

    createTrackbar("min dist", win1, &min_dist, 300, controller);
    createTrackbar("min R", win1, &min_r, 300, controller);
    createTrackbar("max R", win1, &max_r, 300, controller);
    createTrackbar("canny threshold 1", win1, &cannyThreshold1, 300, controller);
    createTrackbar("canny threshold 2", win1, &cannyThreshold2, 300, controller);
    createTrackbar("acc threshold", win1, &accThreshold, 300, controller);
    controller(0, nullptr);

    waitKey(0);
    destroyAllWindows();

    return 0;
}