import paddle.fluid as fluid
import config as cfg
from utils import data_reader
from nets.attention_model import attention_eval
from nets.crnn_ctc_model import ctc_eval
from utils.utility import get_ctc_feeder_data, get_attention_feeder_data


def evaluate():
    if cfg.use_model == "crnn_ctc":
        eval = ctc_eval
        get_feeder_data = get_ctc_feeder_data
    else:
        eval = attention_eval
        get_feeder_data = get_attention_feeder_data

    # define network
    evaluator, cost = eval(cfg.data_shape, cfg.num_classes, use_cudnn=cfg.use_gpu)

    # data reader
    test_reader = data_reader.test(prefix_path=cfg.test_prefix, model=cfg.use_model)

    # prepare environment
    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # load init model
    if cfg.persistables_models_path:
        fluid.io.load_persistables(exe, dirname=cfg.persistables_models_path)
        print("Init model from: %s." % cfg.persistables_models_path)

    evaluator.reset(exe)
    count = 0
    for data in test_reader():
        count += 1
        exe.run(fluid.default_main_program(), feed=get_feeder_data(data, place))
    avg_distance, avg_seq_error = evaluator.eval(exe)
    print("Read %d samples; avg_distance: %s; avg_seq_error: %s" % (count, avg_distance, avg_seq_error))


if __name__ == "__main__":
    evaluate()
