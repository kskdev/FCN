# coding:utf-8
import os
from glob import glob

import chainer.links as L
from chainer import datasets, iterators, optimizers, training, serializers
from chainer.training import extensions as ex

from data import Dataset
from model import FCN

out_dir = './Result'
size = (224, 224)
class_num = 40
batch = 8
gpu_id = -1
initial_lr = 0.01
train_epoch = 100
save_model = 'model.npz'

# 入力画像とラベル画像をglobで抽出できるように設定
image_train = './Data/Images/*.png'
label_train = './Data/Labels/*.png'
image_valid = './Data/Images/*.png'
label_valid = './Data/Labels/*.png'

# ファイルパスをソート(入力画像と教師画像のファイル名が統一されているという前提のためペアをソートして合わせている)
# この辺は人によってやり方がそれぞれなので，お好みで
image_train = sorted(glob(image_train))
label_train = sorted(glob(label_train))
image_valid = sorted(glob(image_valid))
label_valid = sorted(glob(label_valid))

# data.pyのDatasetクラスを利用
# 因みに data_set[i][0]でi番目の入力データ，data_set[i][1]でi番目の教師データを取得可能
data_train = Dataset(image_train, label_train, size)
data_valid = Dataset(image_valid, label_valid, size)

# 学習及び評価用のiteratorオブジェクトを生成
# (マルチプロセス化するなら SerialIterator を MultiprocessIterator に変更)
iter_train = iterators.SerialIterator(data_train, batch, repeat=True, shuffle=True)
iter_valid = iterators.SerialIterator(data_valid, batch, repeat=False, shuffle=False)

# モデルオブジェクトを生成
model = L.Classifier(FCN(class_num=class_num))
# パラメータの最適化手法を選択(SGD, MomentumSGD, Adam, etc...)
optim = optimizers.MomentumSGD(lr=initial_lr)
optim.setup(model)

# 学習を進めるためのプロセスを定義(複雑な学習はupdaterも自作するようになるかも)
updater = training.StandardUpdater(iter_train, optim, device=gpu_id)
trainer = training.Trainer(updater, (train_epoch, 'epoch'), out=out_dir)

# --- Extensions --- #
# 個人的にtrainerを用いた学習方法の最大の利点だと思ってる Extension の設定

# 学習中の情報を記録 (最低限使っておきたいextension)
trainer.extend(ex.LogReport())
# 学習率の記録
trainer.extend(ex.observe_lr())
# 評価画像によるモデルの性能評価
trainer.extend(ex.Evaluator(iter_valid, model, device=gpu_id), name='val')
# 学習の様子を画面に出力(正確にはPrintReport()の引数のoutに合わせて出力)
lst = ['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'elapsed_time', 'lr']
trainer.extend(ex.PrintReport(lst))
# 学修の進捗をプログレスバーで表現
trainer.extend(ex.ProgressBar())
# 学習のLossを可視化
trainer.extend(ex.PlotReport(['main/loss', 'val/main/loss'], 'epoch', file_name='loss.png'))
# 学習のAccuracyを可視化
trainer.extend(ex.PlotReport(['main/accuracy', 'val/main/accuracy'], 'epoch', file_name='accuracy.png'))

# 学習ループの実行
print('train loop is ready...')
trainer.run()

# 学習が終わったらその時のネットワークのパラメータを保存する
# 保存は数Epochごとにモデルを保存するextensionのsnapshot等を使うといい
serializers.save_npz(os.path.join(out_dir, save_model), model)
