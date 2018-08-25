# coding:utf-8

import glob
import os

import cv2  # 他のファイルがPILで統一してたはずなので統一した方がいいかも
import numpy as np


class Convert:
    def __init__(self, size):
        self.size = size
        self.arr_4ch = np.zeros((self.size[1], self.size[0], 4), dtype=np.uint8)
        self.label_id_rgb = np.array(
            # Label_ID, R,G,B
            [[0, 0, 0, 0],
             [1, 85, 0, 0],
             [2, 170, 0, 0],
             [3, 255, 0, 0],
             [4, 0, 85, 0],
             [5, 85, 85, 0],
             [6, 170, 85, 0],
             [7, 255, 85, 0],
             [8, 0, 170, 0],
             [9, 85, 170, 0],
             [10, 170, 170, 0],
             [11, 255, 170, 0],
             [12, 0, 255, 0],
             [13, 85, 255, 0],
             [14, 170, 255, 0],
             [15, 255, 255, 0],
             [16, 0, 0, 85],
             [17, 85, 0, 85],
             [18, 170, 0, 85],
             [19, 255, 0, 85],
             [20, 0, 85, 85],
             [21, 85, 85, 85],
             [22, 170, 85, 85],
             [23, 255, 85, 85],
             [24, 0, 170, 85],
             [25, 85, 170, 85],
             [26, 170, 170, 85],
             [27, 255, 170, 85],
             [28, 0, 255, 85],
             [29, 85, 255, 85],
             [30, 170, 255, 85],
             [31, 255, 255, 85],
             [32, 0, 0, 170],
             [33, 85, 0, 170],
             [34, 170, 0, 170],
             [35, 255, 0, 170],
             [36, 0, 85, 170],
             [37, 85, 85, 170],
             [38, 170, 85, 170],
             [39, 255, 85, 170],
             [40, 0, 170, 170]],
            dtype=np.uint8)

    # ----------------------------
    # LABEL(1ch) to BGR(3ch)
    # ----------------------------
    def label_to_color(self, label_arr):
        '''
        PILとOpenCVの画像読み込み、書き出し辺りのスピードを調査すること
        for文を置き換えれないか考えること
        '''
        self.arr_4ch[:, :, 0] = label_arr
        for i in np.unique(label_arr):
            self.arr_4ch[np.where(self.arr_4ch[:, :, 0] == i)] = self.label_id_rgb[i]
        return self.arr_4ch[:, :, 1:4]

    # ----------------------------
    # BGR(3ch) to LABEL(1ch)
    # ----------------------------
    # TODO まだ途中
    def color_to_label(self, rgb_arr):
        self.arr_4ch[:, :, 1:4] = rgb_arr

        def unique_2d(a):
            dtype1 = np.dtype((np.void, a.dtype.itemsize * np.prod(a.shape[1:])))
            b = np.ascontiguousarray(a.reshape(a.shape[0], -1)).view(dtype1)
            return a[np.unique(b, return_index=1)[1]]

        lst = unique_2d(np.reshape(rgb_arr, newshape=(-1, 3)))
        for i, c in enumerate(lst):
            self.arr_4ch[np.where(self.arr_4ch[:, :, 1:4] == c)] = self.label_id_rgb[i]
        return self.arr_4ch[:, :, 0]

    # 一応動くやつ (あまり早くから使いたくない(numbaで高速化は確認したがnumba依存はしたくない))
    def color_to_label_slowly(self, bgr_img_arr):
        img = bgr_img_arr.tolist()

        def bgr_to_label_pix(bgr):
            for k, b, g, r in self.label_id_rgb:
                if bgr == [b, g, r]:
                    return k
            return 0

        img = [tuple(map(bgr_to_label_pix, img[i])) for i in range(len(img))]
        return np.asarray(img, dtype=np.uint8)


if __name__ == '__main__':
    paths = glob.glob("./image/pred*.png")
    out_dir = './out'
    Size = (640, 480)

    os.makedirs(out_dir, exist_ok=False)
    cvt = Convert(size=Size)
    for f in paths:
        lbl = cv2.resize(cv2.imread(f, 0), dsize=Size)
        rgb = cvt.label_to_color(lbl)
        save_file = f.split('/')[-1]
        cv2.imwrite(os.path.join(out_dir, 'color_' + save_file), rgb)


