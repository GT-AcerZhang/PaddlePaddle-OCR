import os
import numpy as np
import paddle.fluid as fluid
from PIL import Image
import config as cfg
import json

place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

[infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=cfg.infer_model_path,
                                                                              executor=exe,
                                                                              model_filename='model.paddle',
                                                                              params_filename='params.paddle')
with open(cfg.dict_path, 'r', encoding='utf-8') as f:
    d = json.loads(f.readlines())

label_dict = dict(zip(d.values(), d.keys()))


def load_image(img_path):
    img = Image.open(img_path).convert('L')
    # 固定图片的高
    img = img.resize((img.size[0], cfg.data_shape[1]))
    img = np.array(img) - 127.5
    # (batch, channel, width, height)
    img = img[np.newaxis, np.newaxis, ...]
    return img


def prune(words, sos, eos):
    # 删除预测结果中未使用的令牌
    start_index = 0
    end_index = len(words)
    if sos in words:
        start_index = np.where(words == sos)[0][0] + 1
    if eos in words:
        end_index = np.where(words == eos)[0][0]
    return words[start_index:end_index]


def inference(image_path):
    img = load_image(image_path)
    pixel_tensor = fluid.LoDTensor()
    pixel_data = np.concatenate([img], axis=0).astype("float32")
    pixel_tensor.set(pixel_data, place)
    results = exe.run(infer_program,
                feed={feeded_var_names[0]: pixel_tensor},
                fetch_list=target_var,
                return_numpy=False)
    indexes = prune(np.array(results[0]).flatten(), 0, 1)
    r = ''
    for index in indexes:
        r += label_dict[index]
    return r


if __name__ == "__main__":
    path = 'dataset/images'
    images = os.listdir(path)
    for image in images:
        result = inference(os.path.join(path, image))
        print("%s: %s" % (image, result))
