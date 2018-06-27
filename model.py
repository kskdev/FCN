import chainer
import chainer.links as L
import chainer.functions as F


class FCN(chainer.Chain):
    def __init__(self, class_num=41):
        super(FCN, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(3, 64, 3, stride=1, pad=1)
            self.conv1_2 = L.Convolution2D(64, 64, 3, stride=1, pad=1)

            self.conv2_1 = L.Convolution2D(64, 128, 3, stride=1, pad=1)
            self.conv2_2 = L.Convolution2D(128, 128, 3, stride=1, pad=1)

            self.conv3_1 = L.Convolution2D(128, 256, 3, stride=1, pad=1)
            self.conv3_2 = L.Convolution2D(256, 256, 3, stride=1, pad=1)
            self.conv3_3 = L.Convolution2D(256, 256, 3, stride=1, pad=1)

            self.conv4_1 = L.Convolution2D(256, 512, 3, stride=1, pad=1)
            self.conv4_2 = L.Convolution2D(512, 512, 3, stride=1, pad=1)
            self.conv4_3 = L.Convolution2D(512, 512, 3, stride=1, pad=1)

            self.conv5_1 = L.Convolution2D(512, 512, 3, stride=1, pad=1)
            self.conv5_2 = L.Convolution2D(512, 512, 3, stride=1, pad=1)
            self.conv5_3 = L.Convolution2D(512, 512, 3, stride=1, pad=1)

            self.pool3 = L.Convolution2D(256, class_num, 1, stride=1, pad=0)
            self.pool4 = L.Convolution2D(512, class_num, 1, stride=1, pad=0)
            self.pool5 = L.Convolution2D(512, class_num, 1, stride=1, pad=0)

            self.upsample4 = L.Deconvolution2D(class_num, class_num, ksize=4, stride=2, pad=1)
            self.upsample5 = L.Deconvolution2D(class_num, class_num, ksize=8, stride=4, pad=2)
            self.upsample = L.Deconvolution2D(class_num, class_num, ksize=16, stride=8, pad=4)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        p3 = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv4_1(p3))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        p4 = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv5_1(p4))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        p5 = F.max_pooling_2d(h, 2, stride=2)

        p3 = self.pool3(p3)
        p4 = self.upsample4(self.pool4(p4))
        p5 = self.upsample5(self.pool5(p5))

        h = p3 + p4 + p5  # Skip process (FCN-08s)
        out = self.upsample(h)
        return out
