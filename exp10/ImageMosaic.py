import cv2
import numpy as np


def show_image(img, win_name, location_x=0, location_y=0):
    if cv2.getWindowProperty(win_name, cv2.WND_PROP_AUTOSIZE) == -1:
        cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, location_x, location_y)
    cv2.imshow(win_name, img)


def destroy():
    """python这个退出实在是太抽象了"""
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)


def detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[-1] == 3 else img
    surf = cv2.xfeatures2d.SURF_create(2000)
    key_points, descriptor = surf.detectAndCompute(gray, None)
    return key_points, descriptor


def get_matches_homo(kps_left, kps_right, desc_left, desc_right):
    # matcher = FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50}) # opencv3.1中和knnMatch有bug
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    raw_matches = matcher.knnMatch(desc_left, desc_right, 2)

    """
    queryIdx：测试图像的特征点描述符的下标->img_left
    trainIdx：样本图像的特征点描述符下标->img_right
    distance：代表这匹配的特征点描述符的欧式距离，数值越小也就说明俩个特征点越相近。
    """
    matches, good = [], []
    for m, n in raw_matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)
            matches.append((m.queryIdx, m.trainIdx))

    kps_left = np.float32([kp.pt for kp in kps_left])
    kps_right = np.float32([kp.pt for kp in kps_right])
    if len(matches) > 4:
        pts_left = np.float32([kps_left[i] for i, _ in matches])
        pts_right = np.float32([kps_right[j] for _, j in matches])
        """
        注意这里的参数顺序
        cv2.findHomography(pts_right, pts_left, cv2.RANSAC)
        src是right，dst是left，得到的是src到dst的变换矩阵，即是以dst的坐标系为参考坐标系的
        """
        homo, _ = cv2.findHomography(pts_right, pts_left, cv2.RANSAC)
        return good, homo

    return None


def mosaic(img_left, img_right, homo):
    h_left, w_left = img_left.shape[:2]
    h_right, w_right = img_right.shape[:2]
    """
    对img_right进行透视变换
    由于透视变换会改变图片场景的大小，导致部分图片内容看不到
    所以对图片进行扩展，高度取最高的，宽度为两者相加
    """
    img_mosaic = np.zeros((max(h_left, h_right), w_left + w_right, 3), dtype='uint8')
    img_mosaic[0:h_right, 0:w_right] = img_right
    img_mosaic = cv2.warpPerspective(img_mosaic, homo, (img_mosaic.shape[1], img_mosaic.shape[0]))
    img_mosaic[0:h_left, 0:w_left] = img_left
    return img_mosaic


def main():
    img_left = cv2.imread("/home/liyang/图片/left.png", cv2.WINDOW_AUTOSIZE)
    img_right = cv2.imread("/home/liyang/图片/right.png", cv2.WINDOW_AUTOSIZE)

    kps_left, desc_left = detect(img_left)
    kps_right, desc_right = detect(img_right)
    matches, homo = get_matches_homo(kps_left, kps_right, desc_left, desc_right)
    print(homo)

    img_match = cv2.drawMatches(img_left, kps_left, img_right, kps_right, matches, None)
    img_mosaic = mosaic(img_left, img_right, homo)

    show_image(img_match, "match", 200, 200)
    show_image(img_mosaic, "mosaic", 200, 400)
    destroy()


main()
