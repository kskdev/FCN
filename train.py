# coding:utf-8
import os
from glob import glob

import chainer
from chainer import iterators, optimizers, training
from chainer.training import extensions as ex

from data import MyDataSet
from model import FCN

out_dir = './Result'
class_num = 40
size = (224, 224)
batch = 4
train_epoch = 100
gpu_id = 0

# 入力画像とラベル画像をglobで抽出できるように設定
data_root = '/home/DataSet'
image_train = os.path.join(data_root, 'train_images/*.png')
label_train = os.path.join(data_root, 'train_valid/*.png')
image_valid = os.path.join(data_root, 'valid_images/*.png')
label_valid = os.path.join(data_root, 'label_label/*.png')

# ファイルパスをソート(入力画像と教師画像のファイル名が統一されているという前提のためペアをソートして合わせている)
# この辺は人によってやり方がそれぞれなので，お好みで
image_train = sorted(glob(image_train))
label_train = sorted(glob(label_train))
image_valid = sorted(glob(image_valid))
label_valid = sorted(glob(label_valid))

# data.pyのMyDataSetクラスを利用
# 因みに data_set[i][0]でi番目の入力データ，data_set[i][1]でi番目の教師データを取得可能
data_train = MyDataSet(image_train, label_train, size, use_augment=True)
data_valid = MyDataSet(image_valid, label_valid, size, use_augment=False)
print('train data:', len(data_train))
print('valid data:', len(data_valid))

# 学習及び評価用のiteratorオブジェクトを生成
# (マルチプロセス化するなら SerialIterator を MultiprocessIterator に変更)
iter_train = iterators.SerialIterator(data_train, batch, repeat=True, shuffle=True)
iter_valid = iterators.SerialIterator(data_valid, batch, repeat=False, shuffle=False)

# モデルオブジェクトを生成
model = chainer.links.Classifier(FCN(class_num=class_num))
# パラメータの最適化手法を選択(SGD, MomentumSGD, Adam, etc...)
optim = optimizers.Adam()
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
# 学習の進捗をプログレスバーで表現(癒やし)
trainer.extend(ex.ProgressBar(update_interval=1))
# 学習のLossを可視化
trainer.extend(ex.PlotReport(['main/loss', 'val/main/loss'], 'epoch', file_name='loss.png'))
# 学習のAccuracyを可視化
trainer.extend(ex.PlotReport(['main/accuracy', 'val/main/accuracy'], 'epoch', file_name='accuracy.png'))
# 学習のスナップショット(途中で停止してしまった時の保険)
trainer.extend(ex.snapshot(filename='snapshot_epoch_{.updater.epoch}'))
# 学習モデルの保存
trainer.extend(ex.snapshot_object(model, 'model_{.updater.epoch}epoch'))

# 学習ループの実行
print('train loop is ready...')
trainer.run()
