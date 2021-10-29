#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const char *img_path = "/home/liyang/study/cv/Circle Detection Dataset/2.jpg";

Mat img;

bool cmp(Point &a, Point &b) {//点的位置从大到小排序
    if (a.y != b.y) return a.y > b.y;
    return a.x > b.x;
}

void circleDetection(Mat &src, float minDist, int canny_threshold, int acc_threshold) {
    if (src.empty()) CV_Error(Error::StsNullPtr, "no src img");
    int rows = src.rows, cols = src.cols;

    Mat processed;
    src.copyTo(processed);

    Mat src_gray;//转换为灰度图
    cvtColor(src, src_gray, COLOR_RGB2GRAY);

    Mat src_edges;//得到边缘图像
    Canny(src_gray, src_edges, MAX(canny_threshold / 2, 1), canny_threshold);

    Mat dx, dy;//水平梯度和垂直梯度
    Sobel(src_gray, dx, CV_16SC1, 1, 0);
    Sobel(src_gray, dy, CV_16SC1, 0, 1);

    Mat accum = Mat::zeros(rows + 2, cols + 2, CV_32SC1);//累加器矩阵
    vector<Point> nz, centers;//圆周序列和圆心序列
    int max_r = MAX(rows, cols);

    //对边缘图像遍历，计算累加和
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            //如果当前的像素不是边缘点，或者水平梯度值和垂直梯度值都为0，则一定不是圆周上的点
            if (!src_edges.at<uchar>(y, x) || (dx.at<int>(y, x) == 0 && dy.at<int>(y, x) == 0)) continue;

            //计算当前点的梯度值
            auto current_dx = (float) dx.at<int>(y, x), current_dy = (float) dy.at<int>(y, x);
            auto gradient = sqrt(current_dx * current_dx + current_dy * current_dy);
            assert(gradient >= 1);

            //水平和垂直方向的位移量（即梯度方向）
            int step_x = cvRound(current_dx * 0.1f / gradient);
            int step_y = cvRound(current_dy * 0.1f / gradient);

            //在当前点沿着梯度方向和梯度的反方向对经过的像素进行累加
            int x0 = x, y0 = y;
            for (int ori = 0; ori < 2; ori++) {//两个方向
                int x1 = x0 + step_x, y1 = y0 + step_y;
                for (int r = 0; r <= max_r; x1 += step_x, y1 += step_y, r++) {
                    if (x1 > cols - 1 || x1 < 0 || y1 > rows - 1 || y1 < 0) break;
                    accum.at<int>(y1, x1)++;
                }
                //把位移量设置为反方向，下一次就会重新设为正方向
                step_x = -step_x, step_y = -step_y;
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
    minDist *= minDist;//平方，方便比较
    for (auto &p:centers) {
        //使用minDist进行过滤
        int i;
        for (i = 0; i < circles.size(); i++) {
            if (pow(circles[i].x - p.x, 2) + pow(circles[i].y - p.y, 2) < minDist) break;
        }
        if (i < circles.size()) continue;

        //得到候选半径值(保存半径的平方)，这里要注意:没有去重，为了计数
        vector<double> candidate_r2;
        for (auto &p_nz:nz) {//遍历圆周上的点
            double r2 = pow(p_nz.x - p.x, 2) + pow(p_nz.y - p.y, 2);
            if (0 < r2 && r2 <= max_r * max_r) candidate_r2.emplace_back(r2);
        }
        if (candidate_r2.empty()) continue;

        //确定最终半径
        sort(candidate_r2.begin(), candidate_r2.end());
        int start_idx = 0, max_count = 0;
        double r_best, r_pre = sqrt(candidate_r2[0]);
        for (int idx = 1; idx < candidate_r2.size(); idx++) {//NOLINT
            double r = sqrt(candidate_r2[idx]);
            if (r > max_r) break;
            if (r - r_pre < 1) continue;

            //idx-start_idx表示当前得到的相同半径的数量
            //(idx+start_idx)/2表示idx和start_idx中间的数
            //取中间的数所对应的半径值作为当前半径值r_cur，也就是取那些半径值相同的值
            double r_cur = sqrt(candidate_r2[(idx + start_idx) / 2]);
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

    string win2 = "detection";
    namedWindow(win2, WINDOW_AUTOSIZE);
    moveWindow(win2, 800, 200);
    imshow(win2, processed);
}

int main() {
    ios::sync_with_stdio(false);

    img = imread(img_path);

    string win1 = "origin";
    namedWindow(win1, WINDOW_AUTOSIZE);
    moveWindow(win1, 200, 200);
    imshow(win1, img);

    circleDetection(img, (float) img.rows / 20, 100, 60);

    waitKey(0);
    destroyAllWindows();

    return 0;
}