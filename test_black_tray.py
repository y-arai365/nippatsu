import cv2
import numpy as np


class Matrix:
    def __init__(self, upper_left, lower_right):
        self.upper_left = upper_left
        self.lower_right = lower_right

    def get_interval(self, pocket_num):
        """2点とポケット数からintervalを求める"""
        (ul_w, ul_h) = self.upper_left
        (lr_w, lr_h) = self.lower_right
        w_length = lr_w - ul_w
        h_length = lr_h - ul_h
        w_interval = w_length / pocket_num
        h_interval = h_length / pocket_num
        return round(w_interval), round(h_interval)

    def get_upper_left_coordinates(self, pocket_num):
        """座標リストを返す"""
        (ul_w, ul_h) = self.upper_left
        w_interval, h_interval = self.get_interval(pocket_num)
        coordinate_list = []

        for i in range(pocket_num):  # width
            for j in range(pocket_num):  # height
                width = ul_w + (i * w_interval)
                height = ul_h + (j * h_interval)
                coordinate_list.append((width, height))
        return coordinate_list


class Pocket:
    def __init__(self):
        pass

    @staticmethod
    def get_pocket_image(img_orig, coordinate, w_interval, h_interval):
        """取得した画像の内、座標内の範囲を等分した画像を取得する"""
        x, y = round(coordinate[0]), round(coordinate[1])
        return img_orig[y:y + h_interval, x:x + w_interval]

    @staticmethod
    def mark_result(img_orig, coordinate, w_interval, h_interval, result):
        """ポケットに製品があれば緑、なければ赤で描画する"""
        x, y = round(coordinate[0]), round(coordinate[1])
        if result:
            cv2.rectangle(img_orig, (x+4, y+4), (x+w_interval-4, y+h_interval-4), (0, 255, 0), 5)
        else:
            cv2.rectangle(img_orig, (x+4, y+4), (x+w_interval-4, y+h_interval-4), (0, 0, 255), 5)
        return img_orig


class Judgement:
    def __init__(self):
        pass

    def judge(self, pocket_img, th):
        """もらったポケット画像に製品が含まれているか判定"""
        binary_img = self._binarize(pocket_img, th)
        morpho_img = self._morphology(binary_img)
        # width, height = morpho_img.shape[:2]
        # cv2.namedWindow('mor', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('mor', int(height / 4), int(width / 4))
        # cv2.imshow('mor', morpho_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        result = self._judge(morpho_img)
        return result

    @staticmethod
    def _binarize(pocket_img, th):
        """画像を二値化"""
        gray_img = cv2.cvtColor(pocket_img, cv2.COLOR_BGR2GRAY)
        blur_img = cv2.medianBlur(gray_img, 5)
        _, binary_img = cv2.threshold(blur_img, th, 255, cv2.THRESH_BINARY)
        return binary_img

    @staticmethod
    def _morphology(binary_img):
        return cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    @staticmethod
    def _judge(binary_pocket_image):
        """二値化画像に白があれば(製品があれば)True、黒のみなら(製品がなければ)Falseを返す"""
        contours = cv2.findContours(binary_pocket_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for contour in contours:
            area = cv2.contourArea(contour)
            if area:
                return True
            else:
                return False


if __name__ == '__main__':
    # ul_, lr_ = (590, 80), (2960, 2040)
    ul_, lr_ = (600, 80), (2960, 2040)
    pocket_num_ = 10
    th_ = 160

    pers_num_path = "pers_num.npy"
    img_path_ = r"image\original_color\black\101.bmp"
    pts_ = np.load(pers_num_path)
    img_orig_ = cv2.imread(img_path_)

    mat = Matrix(ul_, lr_)
    pocket = Pocket()
    jud = Judgement()
    # pers = PerspectiveTransformer(width_, height_, pts_)

    w_interval_, h_interval_ = mat.get_interval(pocket_num_)
    coordinate_list_ = mat.get_upper_left_coordinates(pocket_num_)
    for coordinate_ in coordinate_list_:
        pocket_img_ = pocket.get_pocket_image(img_orig_, coordinate_, w_interval_, h_interval_)
        result_ = jud.judge(pocket_img_, th_)
        result_img_ = pocket.mark_result(img_orig_, coordinate_, w_interval_, h_interval_, result_)

    width_, height_ = result_img_.shape[:2]
    print(width_, height_)
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('result', int(height_/4), int(width_/4))
    cv2.imshow('result', result_img_)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
