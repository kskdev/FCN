# coding:utf-8

import glob
import os
import os.path as osp
from colorsys import hsv_to_rgb

from PIL import Image
import numpy as np


class Convert:
    def __init__(self):
        self.palette = np.array(
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

    def palette_from_hue(self, classes, background_id=0):
        """
        グレースケール表示を色付けする
        カラーアサインは色相をクラス数で等分し，RGBへ変換したパレットを作成する
        遊びで作ったものなのでカラーアサインは分かりやすいように自分で割り当てるべき
        """
        # generate rgb palette from class_num (value min:0, max:1)
        palette = [hsv_to_rgb(h / classes, 1., 1.) for h in range(classes)]

        # transform palette (x255 and float2int)
        palette = np.asarray(palette, dtype=np.float64) * 255.
        palette = palette.astype(np.uint8)

        # generate class id
        class_ids = np.arange(0, classes, step=1, dtype=np.uint8).reshape((-1, 1))

        # Reset Background ID
        palette[background_id] = np.array([0, 0, 0], dtype=np.uint8)

        # concatenate class_ids and palette
        return np.hstack((class_ids, palette))

    # ----------------------------
    # LABEL(1ch) to BGR(3ch)
    # ----------------------------
    def label_to_color(self, label_arr):
        '''
        遅い
        推論ラベルが予め登録されたラベルと一致しない場合は全て黒色になるので要注意
        '''
        size = label_arr.size
        arr_4ch = np.zeros((size[1], size[0], 4), dtype=np.uint8)
        arr_4ch[:, :, 0] = label_arr

        set_cls = set(self.palette[:, 0])
        set_prd = set(np.unique(label_arr))
        for i in set_prd & set_cls:
            arr_4ch[np.where(arr_4ch[:, :, 0] == i)] = self.palette[i]
        return arr_4ch[:, :, 1:4]

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
            self.arr_4ch[np.where(self.arr_4ch[:, :, 1:4] == c)] = self.palette[i]
        return self.arr_4ch[:, :, 0]

    # 一応動くやつ (あまり早くから使いたくない(numbaで高速化は確認したがnumba依存はしたくない))
    def color_to_label_slowly(self, bgr_img_arr):
        img = bgr_img_arr.tolist()

        def bgr_to_label_pix(bgr):
            for k, b, g, r in self.palette:
                if bgr == [b, g, r]:
                    return k
            return 0

        img = [tuple(map(bgr_to_label_pix, img[i])) for i in range(len(img))]
        return np.asarray(img, dtype=np.uint8)


if __name__ == '__main__':
    paths = glob.glob("./image/pred*.png")
    out_dir = './out'

    os.makedirs(out_dir, exist_ok=True)
    cvt = Convert()

    save_paths = map(lambda path: osp.join(out_dir, 'rgb_' + osp.basename(path)), paths)
    images = map(lambda f: cvt.label_to_color(Image.open(f)), paths)

    for f, img in zip(save_paths, images):
        Image.fromarray(img).save(f)
