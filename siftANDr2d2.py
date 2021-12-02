import argparse
import os
import cv2
import numpy as np


def show_image(img, win_name, location_x=0, location_y=0):
    if cv2.getWindowProperty(win_name, cv2.WND_PROP_AUTOSIZE) == -1:
        cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, location_x, location_y)
    cv2.imshow(win_name, img)


def get_sift_desc(img):
    """
    - Param: image
    - Return: tuple (key points, descriptor)
    """
    sift = cv2.xfeatures2d.SIFT_create(5000)
    key_points, descriptor = sift.detectAndCompute(img, None)
    return key_points, descriptor


def get_r2d2_desc(file):
    """
    - Param: path of r2d2 file
    - Return: tuple (key points, descriptor)
    """
    r2d2_file = np.load(file)
    key_points_arr = r2d2_file["keypoints"]
    key_points = [cv2.KeyPoint(key_points_arr[i][0], key_points_arr[i][1], 1) for i in range(key_points_arr.shape[0])]
    descriptor = r2d2_file["descriptors"]
    return key_points, descriptor


def get_match_img(imgs, key_pts, desc, method):
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    raw_matches = matcher.knnMatch(desc[0], desc[1], 2)
    print("number of raw matches for %s: %d" % (method, len(raw_matches)))

    good = []
    for m, n in raw_matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)
    print("number of good matches for %s: %d" % (method, len(good)))

    img_match = cv2.drawMatches(imgs[0], key_pts[0], imgs[1], key_pts[1], good, None)
    return img_match


def matching(img_names, img_dict, r2d2_dict):
    imgs = []
    key_points_sift, desc_sift = [], []
    key_points_r2d2, desc_r2d2 = [], []

    for img_name in img_names:
        imgs.append(img_dict[img_name])
        temp1, temp2 = get_sift_desc(img_dict[img_name])
        temp3, temp4 = get_r2d2_desc(r2d2_dict[img_name])
        key_points_sift.append(temp1)
        desc_sift.append(temp2)
        key_points_r2d2.append(temp3)
        desc_r2d2.append(temp4)

    sift_match = get_match_img(imgs, key_points_sift, desc_sift, "SIFT")
    r2d2_match = get_match_img(imgs, key_points_r2d2, desc_r2d2, "r2d2")

    return sift_match, r2d2_match


def main():
    assert os.path.exists(args.img_path)
    assert os.path.exists(args.r2d2_path)

    img_name = args.add_img[:2]  # 暂时仅限两张图片
    img_path = [os.path.join(args.img_path, img) for img in img_name]

    img_dict, r2d2_dict = {}, {}
    for i in range(len(img_name)):
        img_dict[img_name[i]] = cv2.imread(img_path[i], cv2.WINDOW_AUTOSIZE)
        r2d2_path = img_path[i] + ".r2d2"
        if not os.path.exists(r2d2_path):
            command = "python %s --model %s --images %s --top-k 5000" % (
                os.path.join(args.r2d2_path, "extract.py"), os.path.join(args.r2d2_path, "models/r2d2_WASF_N16.pt"),
                img_path[i])
            os.system(command)
        r2d2_dict[img_name[i]] = img_path[i] + ".r2d2"

    sift_match, r2d2_match = matching(img_name, img_dict, r2d2_dict)
    show_image(sift_match, "sift", 200, 200)
    show_image(r2d2_match, "r2d2", 200, 600)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default="/home/liyang/图片")
    parser.add_argument("--r2d2_path", type=str, default="/home/liyang/study/cv/r2d2")
    parser.add_argument("--add_img",
                        type=str,
                        action="append",
                        default=["sunflower.png", "sunflower_original.png"],
                        help="add images, such as --add_img=1.png --add_img=2.png will add two images to list.")
    args = parser.parse_args()
    print(args)

    main()
