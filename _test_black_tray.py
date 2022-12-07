import cv2
import numpy as np


class ProcessImage:
    def __init__(self, kernel):
        self.kernel = kernel

    def get_binary_image_from_img_bgr(self, img_bgr):
        """
        bgr画像から二値化反転画像を返す

        Args:
            img_bgr (img_bgr): 射影変換済みの画像

        Returns:
            img_th: 二値化画像

        """
        img_gray = self._grayscale_image(img_bgr)
        img_blur = self._blur_image(img_gray, self.kernel)
        return self._binarize_and_invert_image(img_blur)

    @staticmethod
    def _grayscale_image(img_bgr):
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _blur_image(img_gray, kernel):
        return cv2.medianBlur(img_gray, kernel)

    @staticmethod
    def _binarize_and_invert_image(img_blur):
        # img_bi = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 3)
        _, img_bi = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # _, img_bi = cv2.threshold(img_blur, 100, 255, cv2.THRESH_BINARY_INV)
        print(_)
        w, h = img_bi.shape[:2]
        cv2.namedWindow('_bi', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('_bi', int(h / 4), int(w / 4))
        cv2.imshow('_bi', img_bi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return img_bi

    @staticmethod
    def black_mask(img_bgr, bgr, thresh):
        # 色の閾値
        min_bgr = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
        max_bgr = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])
        # 画像の2値化
        mask_bgr = cv2.inRange(img_bgr, min_bgr, max_bgr)
        # 画像のマスク（合成）
        result_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_bgr)
        return mask_bgr

    @staticmethod
    def opening(img_th):
        return cv2.morphologyEx(img_th, cv2.MORPH_OPEN, np.zeros((4, 4), np.uint8))


class CutOutImage:
    def __init__(self):
        pass

    def get_rot_cut_image_from_binary_image(self, img_bi, img_pers):
        """
        二値化画像から輪郭を取得して、切り取る

        Args:
            img_bi (img_th): 射影変換後に二値化した画像
            img_pers (img_bgr): img_bgr (img_bgr): 射影変換後の画像

        Returns:
            img_bgr: 矩形に沿って切り出して、向きを整えた画像
        """
        _, box = self._get_tray(img_bi)
        return self.trim_box_coordinate(box, img_pers)

        # return self._draw_contours(img_pers, cnt)
        #
        # center, size, deg = self._get_rect(cnt)
        # return self._rot_cut(img_pers, deg, center, size)

    @staticmethod
    def _draw_contours(img, cnt):
        return cv2.drawContours(img, cnt, -1, color=(0, 0, 255), thickness=2)

    def _get_tray(self, img_bi):
        contours, hierarchy = cv2.findContours(img_bi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(filter(lambda x: cv2.contourArea(x) > 1000, contours))
        cnt = self._get_rect_contour(contours)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        print(box)
        return cnt, box

    @staticmethod
    def _get_rect_contour(contours):
        """全輪郭から矩形部分の輪郭を面積の大きさで判断して取得する"""
        len_cnt_list = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            len_cnt_list.append(area)
        max_len_cnt_index = len_cnt_list.index(max(len_cnt_list))
        return contours[max_len_cnt_index]

    # @staticmethod
    # def _get_rect(contour):
    #     """輪郭点からその図形の中心、大きさ、傾きを取得する"""
    #     center, size, deg = cv2.minAreaRect(contour)
    #     size = np.int0(size)
    #     return center, size, deg
    #
    # @staticmethod
    # def _rot_cut(src_img, deg, center, size):
    #     """傾きを整えて画像の切り取りを行う"""
    #     rot_mat = cv2.getRotationMatrix2D(center, deg, 1.0)
    #     rot_mat[0][2] += -center[0] + size[0] / 2  # -(元画像内での中心位置)+(切り抜きたいサイズの中心)
    #     rot_mat[1][2] += -center[1] + size[1] / 2  # 同上
    #     return cv2.warpAffine(src_img, rot_mat, size)

    @staticmethod
    def trim_box_coordinate(box, img):
        upper_left = (max(min(box[:, 0]), 0), max(min(box[:, 1]), 0))
        lower_right = (max(box[:, 0]), max(box[:, 1]))
        return img[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]


class Matrix:
    def __init__(self):
        pass

    @staticmethod
    def get_interval(ul, lr, pocket_num):
        """2点とポケット数からintervalを求める"""
        (ul_w, ul_h) = ul
        (lr_w, lr_h) = lr
        w_length = lr_w - ul_w
        h_length = lr_h - ul_h
        w_interval = w_length / pocket_num
        h_interval = h_length / pocket_num
        return round(w_interval), round(h_interval)

    def get_upper_left_coordinates(self, ul, lr, pocket_num):
        """座標リストを返す"""
        (ul_w, ul_h) = ul
        w_interval, h_interval = self.get_interval(ul, lr, pocket_num)
        coordinate_list = []

        for i in range(pocket_num):  # width
            for j in range(pocket_num):  # height
                width = ul_w + (i * w_interval)
                height = ul_h + (j * h_interval)
                coordinate_list.append((width, height))
        return coordinate_list


class Judgement:
    def __init__(self):
        pass

    def mark_blank_pocket(self, img, coordinate_list, w_interval, h_interval, th):
        img_copy = img.copy()
        result_count = 0
        for coordinate in coordinate_list:
            pocket_image = self._get_pocket_image(img_copy, coordinate, w_interval, h_interval)
            binary_pocket_image = self._process_img_bgr_to_img_th(pocket_image, th)
            w, h = binary_pocket_image.shape[:2]
            cv2.namedWindow('binary_', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('binary_', int(h / 4), int(w / 4))
            cv2.imshow('binary_', binary_pocket_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            has_single_piece = self._judge(binary_pocket_image)
            if not has_single_piece:
                img_copy = self._mark(img_copy, coordinate, w_interval, h_interval)
                result_count += 1
            else:
                img_copy = self._mark2(img_copy, coordinate, w_interval, h_interval)
        print("result_count:", result_count)
        return img_copy, result_count

    @staticmethod
    def _get_pocket_image(img, coordinate, w_interval, h_interval):
        """座標と間隔で画像を分割"""
        x, y = round(coordinate[0]), round(coordinate[1])
        return img[y:y+h_interval, x:x+w_interval]

    @staticmethod
    def _process_img_bgr_to_img_th(img, th):
        """BGR画像を二値化画像に変更する"""
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.medianBlur(img_gray, 5)
        _, binary_pocket_image = cv2.threshold(img_blur, th, 255, cv2.THRESH_BINARY)
        binary_pocket_image = cv2.morphologyEx(binary_pocket_image, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return binary_pocket_image

    @staticmethod
    def _judge_(binary_pocket_image):
        """二値化画像に白が含まれていたら255、黒のみなら0を返す"""
        contours = cv2.findContours(binary_pocket_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if contours:
            contour = np.vstack(contours)
            print(contours, "contours")
            print(contour, "contour")
            area = cv2.contourArea(contour)
            # return int(round(area / 45)) == 1
            print(area)

            width, height = binary_pocket_image.shape[:2]
            cv2.namedWindow('th_img', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('th_img', int(height), int(width))
            cv2.imshow('th_img', binary_pocket_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if np.amax(binary_pocket_image) == 255:
            return True
        else:
            return False
        # return np.amax(binary_pocket_image) == 255

    @staticmethod
    def _judge(binary_pocket_image):
        """二値化画像に白が含まれていたら255、黒のみなら0を返す"""
        contours = cv2.findContours(binary_pocket_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for contour in contours:
            area = cv2.contourArea(contour)
            print(area)
            # width, height = binary_pocket_image.shape[:2]
            # cv2.namedWindow('th_img', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('th_img', int(height), int(width))
            # cv2.imshow('th_img', binary_pocket_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            if area:
                return True
            else:
                return False

    @staticmethod
    def _mark(img, coordinate, w_interval, h_interval):
        """指定範囲に矩形を描画する"""
        x, y = round(coordinate[0]), round(coordinate[1])
        return cv2.rectangle(img, (x+2, y+2), (x+w_interval-2, y+h_interval-2), (0, 0, 255), 10)

    @staticmethod
    def _mark2(img, coordinate, w_interval, h_interval):
        """指定範囲に矩形を描画する"""
        x, y = round(coordinate[0]), round(coordinate[1])
        return cv2.rectangle(img, (x+2, y+2), (x+w_interval-2, y+h_interval-2), (0, 255, 0), 10)


if __name__ == '__main__':
    from perspective_transform import PerspectiveTransformer

    bgr_ = [50, 50, 50]
    thresh_ = 50
    # ul_ = (220, 90)
    # lr_ = (2480, 1980)
    # ul_, lr_ = (460, 230), (2400, 1840)
    ul_, lr_ = (590, 80), (2960, 2040)
    pocket_num_ = 10

    pers_num_path = "pers_num.npy"
    pts_ = np.load(pers_num_path)

    img_path_ = r"image\original_color\black\101.bmp"
    # img_path_ = r"img_pers_24.png"
    img_ = cv2.imread(img_path_)
    height_, width_ = img_.shape[:2]

    pi = ProcessImage(5)
    coi = CutOutImage()
    mat = Matrix()
    jud = Judgement()
    # pers = PerspectiveTransformer(width_, height_, pts_)

    # img_pers_ = pers.transform(img_)

    # img_th_ = pi.get_binary_image_from_img_bgr(img_)
    # img_th_ = pi.black_mask(img_, bgr_, thresh_)
    # img_th_ = pi.opening(img_th_)
    # img_coi_ = coi.get_rot_cut_image_from_binary_image(img_th_, img_)
    w_interval_, h_interval_ = mat.get_interval(ul_, lr_, pocket_num_)
    coordinate_list_ = mat.get_upper_left_coordinates(ul_, lr_, pocket_num_)

    result_image_, result_count_ = jud.mark_blank_pocket(img_, coordinate_list_, w_interval_, h_interval_, 170)
    # result_image_, result_count_ = jud.mark_blank_pocket(img_coi_, coordinate_list_, w_interval_, h_interval_, 170)

    width_, height_ = result_image_.shape[:2]
    print(width_, height_)
    cv2.namedWindow('th_img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('th_img', int(height_/4), int(width_/4))
    cv2.imshow('th_img', result_image_)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
