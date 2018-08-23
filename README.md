# Semantic Segmentation Network (FCN)の実装
chainerのtrainer機能を用いた簡単な実装 <br>
入力画像と教師画像，クラス数とかを合わせれば多分動くはず...<br>
目的はtrainer機能を使うおべんきょーとしてやってみただけ．<br>
故に，ネットワークの説明とかはしない．

trainer を使う利点はトレーニング方法をある程度の規格化とExtensionの利用だと思う．
(プログレスバーを見てるだけで癒される)

READMEが雑だが，編集は面倒だからまたいつの日か


## Environment
- Python:3.5.2 (Anaconda3 4.2.0)

| Library | Version |
|  :---:  |  :---:  |
|cupy     |4.2.0    |
|chainer  |4.2.0    |
バージョンを指定してインストールする場合は`pip install cupy==4.2.0`のように行う．

numpy, pillow(PIL)等が必要だが，Anacondaをインストールしてあればデフォルトで入ってるはず...
<br>
無ければpip install hogeで入るはず...


## Learning
学習はtrain.pyファイルの中身を編集(主にデータセットまでのパスやハイパーパラメータの調整)し，`python train.py` でOK．
<br>
推論結果が欲しければ学習時と同様にpredict.pyを編集し，`python predict.py`を実行すれば良い．
<br>
同様に定量評価もevaluate.pyを編集し，`python evaluate.py`でおｋ
<br>
parse モジュールを使わないのはあんまり好きじゃないから

#### train.py
モデルのフィッティングを行うためのファイル．<br>
一番重要？
#### data.py
画像の読み込みや学習を行うためのデータセットを作るためのモジュール．
#### model.py
ネットワークが定義されたファイル
#### predict.py
train.py で生成したパラメータを用いて推論を行うファイル．<br>
出力される画像の画素値はラベルIDとなっている．
#### evaluate.py
性能評価を定量的に行うためのファイル．<br>
Pixel Accuracy, Class Accuracy, Mean IOUを求める．

## Schedule (気が向いたらいつか更新するリスト)
- 推論結果画像がラベルIDがそのまま画素値になっているため，見づらいのでRGBでカラフルに表現するファイルを追加予定
- FCN 以外のモデルファイルを追加．めんどかったらやらん (SegNet, ICNet, U-Net, PSPNet, DeepLab...)
- Data Augmentation のバリエーションを増やす(かも)
- 入力正規化方法ももう少し検討
- Updaterを StandardUpdaterに任せているので自作Updaterに置き換える
- etc...
