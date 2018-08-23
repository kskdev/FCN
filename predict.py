# coding:utf-8

import glob
import os

from PIL import Image
import numpy as np
import chainer
from chainer import serializers
from chainer.backends import cuda

import data
from model import FCN


def main():
    image_paths = glob.glob('images/*png')
    model_dir = 'Result'
    class_num = 40
    size = (224, 224)
    gpu_id = 0
    model_file = os.path.join(model_dir, 'model_100epoch')
    out_save_dir = os.path.join(model_dir, 'out')

    model = chainer.links.Classifier(FCN(class_num=class_num))
    serializers.load_npz(model_file, model)
    if gpu_id >= 0:
        cuda.get_device_from_id(gpu_id).use()
        model.to_gpu()

    for i, f in enumerate(image_paths):
        # Predict Input Data
        x = data.read_image(f, size)
        x = data.image_norm(x)
        x = cuda.to_gpu(x[np.newaxis, :, :, :])
        y = model.predictor(x).data.argmax(axis=1)[0]

        # Save Predict Image
        input_file_name = f.split('/')[-1]  # get file name
        save_path = os.path.join(out_save_dir, 'pred{:>05}_'.format(i) + input_file_name)
        y = cuda.to_cpu(y.astype(np.uint8))
        Image.fromarray(y).save(save_path)
        print(f, save_path)


if __name__ == '__main__':
    main()
