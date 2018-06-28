# coding:utf-8
from PIL import Image
import numpy as np
import chainer.links as L
from chainer import serializers

import data
from model import FCN

input_image = 'foo.png'
class_num = 40
model_file = 'model.npz'
size = (224, 224)

# モデル(FCN)を生成し，L.Classifierでラップする (最大の目的はPredictorの利用)
model = L.Classifier(FCN(class_num=class_num))
# train.pyで保存した重みパラメータのファイル(.npz形式)をmodelにロードする
serializers.load_npz('model.npz', model)

# 入力画像を読み込む
x = data.read_image(input_image, size)
# 学習時に正規化させた方法と同様の処理を施す
x = data.image_norm(x)
# クラスIDをピクセルごとに出力させる処理(出力データに対してsoftmaxを掛け，argmaxを取ったもの)
y = model.predictor(x[np.newaxis, :, :, :]).data.argmax(axis=1)[0]

# 予測画像として保存
Image.fromarray(y.astype(np.uint8)).save('output.png')
# 予測画像と比較するために入力画像をリサイズして出力
img = Image.open(input_image).resize(size, Image.BILINEAR).save('input.png')

