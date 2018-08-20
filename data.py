from PIL import Image
import numpy as np
import chainer


# Trainer用 データセットの作り方 https://qiita.com/mitmul/items/5502ecdd2f0b444c427f

# 入力画像の読み込み処理の定義 (3ch)
def read_image(image_path, resize):
    img = Image.open(image_path).convert('RGB').resize(resize, Image.BILINEAR)  # img:(width, height, RGB3ch)
    return np.asarray(img).transpose(2, 0, 1).astype(np.float32)  # transposeにより，img:(RGB3ch, width, height)となる


# 入力画像の前処理(正規化等)を定義
def image_norm(image_array):
    return image_array / 255.0


# セグメンテーションを行うため，教師画像を読み込む処理を定義 (1ch)
def read_label(label_path, resize):
    lbl = Image.open(label_path).resize(resize, Image.NEAREST)
    return np.asarray(lbl, dtype=np.int32)


def augment_data(image, label):
    # 50%の確率で画像を左右反転する(Data Augmentationに相当)
    # TODO 挙動がホントに正しい？
    if np.random.rand() > 0.5:
        image = image[..., ::-1]
        label = label[..., ::-1]

    return image, label


# chainerのiteratorにデータを渡すための処理
class Dataset(chainer.dataset.DatasetMixin):
    def __init__(self, images, labels, size=(224, 224), is_train=False):
        '''
        :param images: 入力画像のパスのリスト
        :param labels: 教師画像のパスのリスト
        :param size: (ネットワークに入力する際のリサイズサイズ tupleで与える(width, height))
        '''
        self.image_paths = images
        self.label_paths = labels
        self.size = size

    # chainer.dataset.DatasetMixinからメソッドをオーバライド
    def __len__(self):
        return len(self.image_paths)

    # chainer.dataset.DatasetMixinからメソッドをオーバライド
    def get_example(self, i):
        # 入力画像と教師画像をそれぞれ読み込み
        img = read_image(self.image_paths[i], self.size)
        lbl = read_label(self.label_paths[i], self.size)

        # 入力画像の正規化 (認識精度を上げるための処理)
        img = image_norm(img)

#         # is_trainingはtrainとvalidでデータセットを変えるための処理(Data Augmentation等)
#         if is_train:
#             return augment_data(img, lbl)
#         # 検証用，テスト用の振る舞いをしてほしい場合
#         else:
#             return img, lbl
        return img, lbl
