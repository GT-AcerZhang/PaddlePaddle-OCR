import os
import paddle.fluid as fluid
import config as cfg
from nets.attention_model import attention_infer
from nets.crnn_ctc_model import ctc_infer
from PIL import Image
import numpy as np

# OCR inference
if cfg.use_model == "crnn_ctc":
    infer = ctc_infer
else:
    infer = attention_infer
# 获取网络输入
images = fluid.data(name='pixel', shape=[None] + cfg.data_shape, dtype='float32')
ids = infer(images, cfg.num_classes, use_cudnn=cfg.use_gpu)
# 准备环境
place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())


def load_image(img_path):
    img = Image.open(img_path).convert('L')
    # 固定图片的高
    img = img.resize((img.size[0], cfg.data_shape[1]))
    img = np.array(img) - 127.5
    img = img[np.newaxis, ...]
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


# 加载模型参数
fluid.load(program=fluid.default_main_program(),
           model_path=cfg.init_model,
           executor=exe,
           var_list=fluid.io.get_program_parameter(fluid.default_main_program()))
print("Init model from: %s." % cfg.init_model)

fluid.io.save_inference_model(dirname=cfg.infer_model_path,
                              feeded_var_names=[images.name],
                              target_vars=[ids],
                              executor=exe)
