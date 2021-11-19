#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const char *img_path = "/home/liyang/图片/sunflower.png";
const char *img_template_path = "/home/liyang/图片/sunflower_tongue.png";

Mat src_img, src_gray;
Mat src_template, template_gray;

void mergeImg(vector<Mat> &vec, Mat &dst) {//多张图片合并为一张，一行2张
    int count = (int) vec.size();
    assert(count > 0 && count <= 6);
    int rowScale = cvRound((float) count / 2), colScale = min(2, count);
    int maxCol = 0, maxRow = 0;
    for (auto &img:vec) {
        int col = img.cols / colScale, row = img.rows / colScale;
        resize(img, img, Size(col, row));
        maxCol = max(col, maxCol), maxRow = max(row, maxRow);
    }
    dst.create(maxRow * rowScale, maxCol * colScale, vec[0].type());
    for (int i = 0; i < count; i++) {
        int x = (i % 2) * maxCol, y = (i / 2) * maxRow;
        vec[i].copyTo(dst(Rect(x, y, vec[i].cols, vec[i].rows)));
    }
}

void showImage(Mat &image, const char *windowName, int location_x, int location_y) {
    //当窗口关闭时，getWindowProperty将返回-1
    if (getWindowProperty(windowName, WND_PROP_AUTOSIZE) == -1)
        namedWindow(windowName, WINDOW_AUTOSIZE);
    moveWindow(windowName, location_x, location_y);
    imshow(windowName, image);
}


int main() {
    ios::sync_with_stdio(false);

    src_img = imread(img_path);
    src_template = imread(img_template_path);
    cvtColor(src_img, src_gray, COLOR_BGR2GRAY);//BGR != RGB
    cvtColor(src_template, template_gray, COLOR_BGR2GRAY);
    
    waitKey(0);
    destroyAllWindows();

    return 0;
}