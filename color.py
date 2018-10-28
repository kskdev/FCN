# coding:utf-8

# Standard Modules
import glob
import os
import os.path as osp
from colorsys import hsv_to_rgb

# External Modules
from PIL import Image
import numpy as np


class Assign:
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

    # ----------------------------
    # LABEL(1ch) to RGB(3ch)
    # ----------------------------
    def id2rgb(self, array_1ch):
        # 推論ラベルが予め登録されたラベルと一致しない場合は全て白色にするので要注意

        # Prepare array.
        w, h = array_1ch.shape
        arr_4ch = np.zeros((w, h, 4), dtype=np.uint8) * 255
        arr_4ch[:, :, 0] = array_1ch

        # Assign label.
        unique_label = set(self.palette[:, 0]) & set(np.unique(array_1ch))
        for i in unique_label:
            arr_4ch[arr_4ch[:, :, 0] == i] = self.palette[i]
        return arr_4ch[:, :, 1:4]

    # ----------------------------
    # RGB(3ch) to LABEL(1ch)
    # ----------------------------
    def rgb2id(self, array_3ch):
        # 登録されたラベルと一致しないラベルは全て白色にするので要注意

        # Prepare array.
        w, h = array_3ch.shape[:2]
        arr_2ch = np.zeros((w, h, 2), dtype=np.int32) * 255
        _rgb_arr = np.copy(array_3ch).astype(np.int32) + 1
        r = _rgb_arr[:, :, 0] * 1000000
        g = _rgb_arr[:, :, 1] * 1000
        b = _rgb_arr[:, :, 2] * 1
        arr_2ch[:, :, 1] = r + g + b

        # Re-define Palette.
        _palette = np.copy(self.palette).astype(np.int32) + 1
        r = _palette[:, 1] * 1000000
        g = _palette[:, 2] * 1000
        b = _palette[:, 3] * 1
        _palette = np.vstack((self.palette[:, 0], r + g + b)).T

        # Assign label.
        for i, (_, p) in enumerate(_palette):
            arr_2ch[arr_2ch[:, :, 1] == p] = _palette[i]
        return np.asarray(arr_2ch[:, :, 0], dtype=np.uint8)


def palette_from_hue(classes, background_id=0):
    # Convert ID into RGB using Hue of HSV.

    # generate rgb palette from class_num (value range -> min:0, max:1)
    palette = [hsv_to_rgb(h / classes, 1., 1.) for h in range(classes)]

    # transform palette
    palette = np.asarray(palette, dtype=np.float32) * 255.
    palette = palette.astype(np.uint8)

    # generate class id
    class_ids = np.arange(0, classes, step=1, dtype=np.uint8).reshape((-1, 1))

    # Reset Background ID
    palette[background_id] = np.array([0, 0, 0], dtype=np.uint8)

    # concatenate class_ids and palette
    return np.hstack((class_ids, palette))


def example_id2rgb():
    paths = glob.glob('./image/label*.png')
    out_dir = './out'

    images = map(lambda p: np.asarray(Image.open(p).convert('L')), paths)
    converted_images = map(lambda a: Assign().id2rgb(a), images)

    save_paths = map(lambda p: osp.join(out_dir, 'rgb_' + osp.basename(p)), paths)

    os.makedirs(out_dir, exist_ok=True)
    for f, img in zip(save_paths, converted_images):
        Image.fromarray(img).save(f)


def example_rgb2id():
    paths = glob.glob("out/*.png")
    out_dir = './seg'

    images = map(lambda p: np.asarray(Image.open(p).convert('RGB')), paths)
    converted_images = map(lambda a: Assign().rgb2id(a), images)

    save_paths = map(lambda p: osp.join(out_dir, 'id_' + osp.basename(p)), paths)

    os.makedirs(out_dir, exist_ok=True)
    for f, img in zip(save_paths, converted_images):
        Image.fromarray(img).save(f)


if __name__ == '__main__':
    example_id2rgb()
    example_rgb2id()
