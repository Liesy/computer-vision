#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const char *img_path_original = "/home/liyang/图片/sunflower_original.png";
const char *img_path_scale = "/home/liyang/图片/sunflower_scale.png";
const char *img_path_local = "/home/liyang/图片/sunflower.png";

Mat src_img_original, src_gray_original;
Mat src_img_scale, src_gray_scale;
Mat src_img_local, src_gray_local;

vector<KeyPoint> KP_original, KP_scale, KP_local;
Mat desc_original, desc_scale, desc_local;

double t, t1, t2;

void showImage(Mat &image, const char *windowName, int location_x, int location_y) {
    namedWindow(windowName, WINDOW_AUTOSIZE);
    moveWindow(windowName, location_x, location_y);
    imshow(windowName, image);
}

void matching(const char *algorithmName) {
    FlannBasedMatcher scaleMatcher, localMatcher;
    vector<DMatch> scaleMatchPts, localMatchPts;
    scaleMatcher.match(desc_original, desc_scale, scaleMatchPts);
    localMatcher.match(desc_original, desc_local, localMatchPts);

    Mat img_scaleMatch, img_localMatch;
    drawMatches(src_img_original, KP_original, src_img_scale, KP_scale, scaleMatchPts, img_scaleMatch);
    drawMatches(src_img_original, KP_original, src_img_local, KP_scale, localMatchPts, img_localMatch);

//    showImage(img_scaleMatch, algorithmName, 200, 200);
//    showImage(img_localMatch, algorithmName, 400, 200);
}

void featureProcess() {
    Ptr<SIFT> sift = SIFT::create(2000);
    Ptr<ORB> orb = ORB::create(2000);
    t = (double) getTickCount();
    //SIFT特征点提取
    sift->detect(src_gray_original, KP_original);
    sift->detect(src_gray_scale, KP_scale);
    sift->detect(src_gray_local, KP_local);
    //特征点描述
    sift->compute(src_gray_original, KP_original, desc_original);
    sift->compute(src_gray_scale, KP_scale, desc_scale);
    sift->compute(src_gray_local, KP_local, desc_local);
    matching("sift");
    t1 = (double) getTickCount();
    //ORB特征点提取
    orb->detect(src_gray_original, KP_original);
    orb->detect(src_gray_scale, KP_scale);
    orb->detect(src_gray_local, KP_local);
    //特征点描述
    orb->compute(src_gray_original, KP_original, desc_original);
    orb->compute(src_gray_scale, KP_scale, desc_scale);
    orb->compute(src_gray_local, KP_local, desc_local);
    matching("orb");
    t2 = (double) getTickCount();
}

int main() {
    ios::sync_with_stdio(false);

    src_img_original = imread(img_path_original);
    src_img_scale = imread(img_path_scale);
    src_img_local = imread(img_path_local);
    cvtColor(src_img_original, src_gray_original, COLOR_BGR2GRAY);//BGR != RGB
    cvtColor(src_img_scale, src_gray_scale, COLOR_BGR2GRAY);
    cvtColor(src_img_local, src_gray_local, COLOR_BGR2GRAY);

    featureProcess();
    cout << "SIFT finished in " << ((t1 - t) * 1000.0) / getTickFrequency() << "ms" << '\n';
    cout << "ORB finished in " << ((t2 - t1) * 1000.0) / getTickFrequency() << "ms" << '\n';

    waitKey(0);
    destroyAllWindows();

    return 0;
}