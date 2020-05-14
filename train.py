import os
import time
import numpy as np
import paddle.fluid as fluid
import config as cfg
from nets.attention_model import attention_train_net
from nets.crnn_ctc_model import ctc_train_net
from utils import data_reader
from utils.utility import get_ctc_feeder_data, get_attention_feeder_data


def main():
    """OCR training"""
    if cfg.use_model == "crnn_ctc":
        train_net = ctc_train_net
        get_feeder_data = get_ctc_feeder_data
    else:
        train_net = attention_train_net
        get_feeder_data = get_attention_feeder_data

    # define network
    sum_cost, error_evaluator, inference_program, model_average = train_net(cfg, cfg.data_shape, cfg.num_classes)

    # data reader
    train_reader = data_reader.train(batch_size=cfg.batch_size,
                                     prefix_path=cfg.train_prefix,
                                     cycle=cfg.total_step > 0,
                                     model=cfg.use_model)
    test_reader = data_reader.test(prefix_path=cfg.test_prefix, model=cfg.use_model)

    # prepare environment
    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # 加载初始化模型
    if cfg.init_model:
        fluid.load(program=fluid.default_main_program(),
                   model_path=cfg.init_model,
                   executor=exe,
                   var_list=fluid.io.get_program_parameter(fluid.default_main_program()))
        print("Init model from: %s." % cfg.init_model)

    train_exe = exe
    error_evaluator.reset(exe)
    if cfg.parallel:
        train_exe = fluid.ParallelExecutor(use_cuda=cfg.use_gpu, loss_name=sum_cost.name)

    fetch_vars = [sum_cost] + error_evaluator.metrics

    def train_one_batch(data):
        var_names = [var.name for var in fetch_vars]
        if cfg.parallel:
            results = train_exe.run(var_names,
                                    feed=get_feeder_data(data, place))
            results = [np.array(r).sum() for r in results]
        else:
            results = exe.run(program=fluid.default_main_program(),
                              feed=get_feeder_data(data, place),
                              fetch_list=fetch_vars)
            results = [r[0] for r in results]
        return results

    def test():
        error_evaluator.reset(exe)
        for data in test_reader():
            exe.run(inference_program, feed=get_feeder_data(data, place))
        _, test_seq_error = error_evaluator.eval(exe)
        return test_seq_error[0]

    def save_model():
        if not os.path.exists(cfg.model_path):
            os.makedirs(cfg.model_path)
        fluid.save(program=fluid.default_main_program(),
                   model_path=os.path.join(cfg.model_path, "model"))
        print("Saved model to: %s" % cfg.model_path)

    iter_num = 0
    stop = False
    while not stop:
        total_loss = 0.0
        total_seq_error = 0.0
        # train a pass
        for data in train_reader():
            if cfg.total_step < iter_num:
                stop = True
                break
            result = train_one_batch(data)
            total_loss += result[0]
            total_seq_error += result[2]

            iter_num += 1
            # training log
            if iter_num % cfg.log_period == 0:
                print("[%s] - Iter[%d]; Avg loss: %.3f; Avg seq err: %.3f"
                      % (time.asctime(time.localtime(time.time())), iter_num,
                         total_loss / (cfg.log_period * cfg.batch_size),
                         total_seq_error / (cfg.log_period * cfg.batch_size)))
                total_loss = 0.0
                total_seq_error = 0.0

            # evaluate
            if iter_num % cfg.eval_period == 0:
                if model_average:
                    with model_average.apply(exe):
                        test_seq_error = test()
                else:
                    test_seq_error = test()

                print("\n[%s] - Iter[%d]; Test seq error: %.3f\n" %
                      (time.asctime(time.localtime(time.time())), iter_num, test_seq_error))
            # save model
            if iter_num % cfg.save_model_period == 0:
                if model_average:
                    with model_average.apply(exe):
                        save_model()
                else:
                    save_model()


if __name__ == "__main__":
    main()
