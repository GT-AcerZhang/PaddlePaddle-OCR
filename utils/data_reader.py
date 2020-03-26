import mmap
import cv2
import numpy as np
import paddle

import config as cfg

SOS = 0
EOS = 1


class ImageData(object):
    def __init__(self, prefix_path):
        self.offset_dict = {}
        for line in open(prefix_path + '.header', 'rb'):
            key, val_pos, val_len = line.split('\t'.encode('ascii'))
            self.offset_dict[key] = (int(val_pos), int(val_len))
        self.fp = open(prefix_path + '.data', 'rb')
        self.m = mmap.mmap(self.fp.fileno(), 0, access=mmap.ACCESS_READ)
        print('loading label')
        # 获取label
        self.label = {}
        for line in open(prefix_path + '.label', 'rb'):
            key, label = line.split(b'\t')
            self.label[key] = [int(c) for c in label.decode().replace('\n', '').split(',')]
        print('finish loading data:', len(self.label))

    # 获取图像数据
    def get_img(self, key):
        p = self.offset_dict.get(key, None)
        if p is None:
            return None
        val_pos, val_len = p
        return self.m[val_pos:val_pos + val_len]

    # 获取图像标签
    def get_label(self, key):
        return self.label.get(key)

    # 获取所有keys
    def get_keys(self):
        return self.label.keys()


class DataGenerator(object):
    def __init__(self, model="crnn_ctc"):
        self.model = model

    def train_reader(self, prefix_path, batchsize, cycle):
        def reader():
            imageData = ImageData(prefix_path)
            keys = imageData.get_keys()
            keys = list(keys)
            np.random.shuffle(keys)

            sizes = len(keys) // batchsize
            if sizes == 0:
                raise ValueError('Batch size is bigger than the dataset size.')
            while True:
                for i in range(sizes):
                    result = []
                    sz = [0, 0]
                    for j in range(batchsize):
                        img = imageData.get_img(keys[i])
                        assert (img is not None)
                        label = imageData.get_label(keys[i])
                        assert (label is not None)

                        img = np.fromstring(img, dtype=np.uint8)
                        img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
                        assert (img is not None), 'image is None'

                        if j == 0:
                            sz = img.shape
                        img = cv2.resize(img, (sz[0], cfg.data_shape[1]))
                        img = np.array(img) - 127.5
                        img = img[np.newaxis, ...]
                        if self.model == "crnn_ctc":
                            result.append([img, label])
                        else:
                            result.append([img, [SOS] + label, label + [EOS]])
                    yield result
                if not cycle:
                    break

        return reader

    def test_reader(self, prefix_path):

        def reader():
            imageData = ImageData(prefix_path)
            keys = imageData.get_keys()
            keys = list(keys)

            for key in keys:
                img = imageData.get_img(key)
                assert (img is not None)
                label = imageData.get_label(key)
                assert (label is not None)

                img = np.frombuffer(img, dtype=np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
                assert (img is not None), 'image is None'

                img = cv2.resize(img, (img.shape[0], cfg.data_shape[1]))
                img = np.array(img) - 127.5
                img = img[np.newaxis, ...]
                if self.model == "crnn_ctc":
                    yield img, label
                else:
                    yield img, [SOS] + label, label + [EOS]

        return reader


def train(batch_size, prefix_path, model="crnn_ctc", cycle=True):
    generator = DataGenerator(model)
    return generator.train_reader(prefix_path, batch_size, cycle)


def test(prefix_path, batch_size=1, model="crnn_ctc"):
    generator = DataGenerator(model)
    return paddle.batch(generator.test_reader(prefix_path), batch_size)
