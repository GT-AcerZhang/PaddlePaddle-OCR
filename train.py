import os
import time
import shutil
import numpy as np
import paddle.fluid as fluid
import config as cfg
from nets.attention_model import attention_train_net
from nets.crnn_ctc_model import ctc_train_net
from utils import data_reader
from visualdl import LogWriter
from utils.utility import get_ctc_feeder_data, get_attention_feeder_data

# 创建记录器
log_writer = LogWriter(dir='log/', sync_cycle=10)

# 创建训练和测试记录数据工具
with log_writer.mode('train') as writer:
    train_loss_writer = writer.scalar('Avg loss')
    train_error_writer = writer.scalar('Avg seq err')

with log_writer.mode('test') as writer:
    test_error_writer = writer.scalar('Seq error')


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
    if cfg.pretrained_model:
        def if_exist(var):
            if os.path.exists(os.path.join(cfg.pretrained_model, var.name)):
                print('loaded: %s' % var.name)
            return os.path.exists(os.path.join(cfg.pretrained_model, var.name))

        fluid.io.load_vars(exe, cfg.pretrained_model, predicate=if_exist)

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
            results = [np.array(result).sum() for result in results]
        else:
            results = train_exe.run(feed=get_feeder_data(data, place),
                                    fetch_list=fetch_vars)
            results = [result[0] for result in results]
        return results

    def test():
        error_evaluator.reset(exe)
        for data in test_reader():
            exe.run(inference_program, feed=get_feeder_data(data, place))
        _, test_seq_error = error_evaluator.eval(exe)
        return test_seq_error[0]

    def save_model(exe):
        if os.path.exists(cfg.persistables_models_path):
            shutil.rmtree(cfg.persistables_models_path)
        else:
            os.makedirs(cfg.persistables_models_path)
        fluid.io.save_persistables(exe, dirname=cfg.persistables_models_path)
        print("Saved model to: %s" % cfg.persistables_models_path)

    iter_num = 0
    train_step = 0
    test_step = 0
    stop = False
    while not stop:
        total_loss = 0.0
        total_seq_error = 0.0
        # train a pass
        for data in train_reader():
            if cfg.total_step < iter_num:
                stop = True
                break
            results = train_one_batch(data)
            total_loss += results[0]
            total_seq_error += results[2]

            iter_num += 1
            # training log
            if iter_num % cfg.log_period == 0:
                print("[%s] - Iter[%d]; Avg loss: %.3f; Avg seq err: %.3f"
                      % (time.asctime(time.localtime(time.time())), iter_num,
                         total_loss / (cfg.log_period * cfg.batch_size),
                         total_seq_error / (cfg.log_period * cfg.batch_size)))

                # 写入训练输出数据
                train_loss_writer.add_record(train_step, total_loss / (cfg.log_period * cfg.batch_size))
                train_error_writer.add_record(train_step, total_seq_error / (cfg.log_period * cfg.batch_size))
                train_step += 1

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
                # 写入测试输出数据
                test_error_writer.add_record(test_step, test_seq_error)
                test_step += 1

            # save model
            if iter_num % cfg.save_model_period == 0:
                if model_average:
                    with model_average.apply(exe):
                        save_model(exe)
                else:
                    save_model(exe)


if __name__ == "__main__":
    main()
