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

void showImage(Mat &image, const char *windowName, int location_x, int location_y) {
    namedWindow(windowName, WINDOW_AUTOSIZE);
    moveWindow(windowName, location_x, location_y);
    imshow(windowName, image);
}

void siftMatching() {
    double start, end;

    //SIFT特征点提取
    Ptr<FeatureDetector> siftDetector = SIFT::create(400);
    vector<KeyPoint> KP_original, KP_scale, KP_local;
    start = (double) getTickCount();
    siftDetector->detect(src_gray_original, KP_original);
    siftDetector->detect(src_gray_scale, KP_scale);
    siftDetector->detect(src_gray_local, KP_local);
    end = (double) getTickCount();
    cout << "SIFT finished in " << ((end - start) * 1000.0) / getTickFrequency() << "ms" << '\n';

    //特征点描述
    Ptr<DescriptorExtractor> siftDescriptor = SIFT::create(2000);
    Mat desc_original, desc_scale, desc_local;
    siftDescriptor->compute(src_gray_original, KP_original, desc_original);
    siftDescriptor->compute(src_gray_scale, KP_scale, desc_scale);
    siftDescriptor->compute(src_gray_local, KP_local, desc_local);

    //获得匹配特征点
    FlannBasedMatcher scaleMatcher, localMatcher;
    vector<DMatch> scaleMatchPts, localMatchPts;
    scaleMatcher.match(desc_original, desc_scale, scaleMatchPts);
    localMatcher.match(desc_original, desc_local, localMatchPts);

    Mat img_scaleMatch, img_localMatch;
    drawMatches(src_img_original, KP_original, src_img_scale, KP_scale, scaleMatchPts, img_scaleMatch);
    drawMatches(src_img_original, KP_original, src_img_local, KP_local, localMatchPts, img_localMatch);

    showImage(img_scaleMatch, "sift scale matching", 200, 200);
    showImage(img_localMatch, "sift local matching", 800, 200);
}

void orbMatching() {
    double start, end;

    //SIFT特征点提取
    Ptr<FeatureDetector> orbDetector = ORB::create(400);
    vector<KeyPoint> KP_original, KP_scale, KP_local;
    start = (double) getTickCount();
    orbDetector->detect(src_gray_original, KP_original);
    orbDetector->detect(src_gray_scale, KP_scale);
    orbDetector->detect(src_gray_local, KP_local);
    end = (double) getTickCount();
    cout << "orb finished in " << ((end - start) * 1000.0) / getTickFrequency() << "ms" << '\n';

    //特征点描述
    Ptr<DescriptorExtractor> orbDescriptor = ORB::create(2000);
    Mat desc_original, desc_scale, desc_local;
    orbDescriptor->compute(src_gray_original, KP_original, desc_original);
    orbDescriptor->compute(src_gray_scale, KP_scale, desc_scale);
    orbDescriptor->compute(src_gray_local, KP_local, desc_local);

    //获得匹配特征点
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    vector<DMatch> scaleMatchPts, localMatchPts;
    matcher->match(desc_original, desc_scale, scaleMatchPts);
    matcher->match(desc_original, desc_local, localMatchPts);

    Mat img_scaleMatch, img_localMatch;
    drawMatches(src_img_original, KP_original, src_img_scale, KP_scale, scaleMatchPts, img_scaleMatch);
    drawMatches(src_img_original, KP_original, src_img_local, KP_local, localMatchPts, img_localMatch);

    showImage(img_scaleMatch, "orb scale matching", 200, 200);
    showImage(img_localMatch, "orb local matching", 800, 200);
}

int main() {
    ios::sync_with_stdio(false);

    src_img_original = imread(img_path_original);
    src_img_scale = imread(img_path_scale);
    src_img_local = imread(img_path_local);
    cvtColor(src_img_original, src_gray_original, COLOR_BGR2GRAY);//BGR != RGB
    cvtColor(src_img_scale, src_gray_scale, COLOR_BGR2GRAY);
    cvtColor(src_img_local, src_gray_local, COLOR_BGR2GRAY);

    siftMatching();
    orbMatching();

    waitKey(0);
    destroyAllWindows();

    return 0;
}