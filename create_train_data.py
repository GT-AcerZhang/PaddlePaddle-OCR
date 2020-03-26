import os
from tqdm import tqdm
import config
import random
from utils.data_format_converter import convert_data


# 打乱数据
def shuffle_data(data_list_path):
    with open(data_list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        random.shuffle(lines)
    with open(data_list_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)


# 生成图像列表
def create(images_dir, train_list_path, test_list_path, label_file):
    f_train = open(train_list_path, 'w', encoding='utf-8')
    f_test = open(test_list_path, 'w', encoding='utf-8')
    f_label = open(label_file, 'w', encoding='utf-8')
    label_dict = dict()
    images = os.listdir(images_dir)
    i = 0
    for image in tqdm(images):
        i += 1
        image_path = os.path.join(images_dir, image)
        chars = image.split('_')[1]
        for c in chars:
            keys = label_dict.keys()
            if c not in keys:
                label_dict[c] = len(label_dict) + 1
        labels = [label_dict.get(c) for c in chars]

        label = ''
        for l in labels:
            label += '%s,' % l
        label = label[:-1]
        if i % 20 == 0:
            f_test.write("%s %s\n" % (image_path.replace('\\', '/'), label))
        else:
            f_train.write("%s %s\n" % (image_path.replace('\\', '/'), label))

    f_label.write("%s\n" % str(label_dict).replace("'", '"'))
    f_train.close()
    f_test.close()
    f_label.close()

    shuffle_data(train_list_path)
    print('create data list done!')


if __name__ == '__main__':
    create('dataset/images', config.train_list, config.test_list, config.dict_path)
    convert_data(config.train_list, config.train_prefix)
    convert_data(config.test_list, config.test_prefix)
