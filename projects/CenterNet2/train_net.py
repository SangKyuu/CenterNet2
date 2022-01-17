import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import time
import datetime
import json
import cv2
import re
import argparse
import sys

from fvcore.common.timer import Timer
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    DatasetCatalog
)
from detectron2.structures import BoxMode

from detectron2.engine import default_argument_parser, default_setup, launch

from detectron2.evaluation import (
    COCOEvaluator,
    LVISEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.modeling.test_time_augmentation import GeneralizedRCNNWithTTA
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.build import build_detection_train_loader

from centernet.config import add_centernet_config
from centernet.data.custom_build_augmentation import build_custom_augmentation

logger = logging.getLogger("detectron2")


def ratio_convert(cord, width, height):
    return cord[0] * width, cord[1] * height, cord[2] * width, cord[3] * height


def c_xywh2xyxy(x, y, w, h):
    return x - 0.5 * w, y - 0.5 * h, x + 0.5 * w, y + 0.5 * h


def preprocess_txt2coco(txt_dir):
    with open(txt_dir, 'r') as f:  # , -1,"utf-8"
        img_dirs = f.readlines()

    dir = '/data/Paper_dataset/'

    dataset_dicts = []
    for idx, img_dir in enumerate(img_dirs):
        record = {}
        img_dir = img_dir.rstrip('\n')
        # img_dir = img_dir.replace('/data/Paper_dataset/',dir)
        # img_dir = dir + img_dir
        # img_dir = dir + img_dir.lstrip('./')

        label_file = re.sub(r'.jpg$|.png$|.JPG$', '.txt',
                            img_dir)  # img_dir.replace('.jpg', '.txt').replace('images', 'labels')
        with open(label_file, 'r') as f:
            labels = f.readlines()

        # assert len(labels) > 0

        height, width = cv2.imread(img_dir).shape[:2]

        record["file_name"] = img_dir
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []
        for anno in labels:
            anno = anno.rstrip('\n')
            c, x, y, w, h = map(float, anno.split(' '))
            # x1, y1, x2, y2, c = map(float, anno.split(' '))
            obj = {
                "bbox": list(c_xywh2xyxy(*ratio_convert([x, y, w, h], width, height))),
                # "bbox": [x1, y1, x2, y2],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": int(c),  # 0
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def do_test(cfg, model):
    results = OrderedDict()

    for dataset_name in cfg.DATASETS.TEST:
        mapper = None if cfg.INPUT.TEST_INPUT_TYPE == 'default' else \
            DatasetMapper(
                cfg, False, augmentations=build_custom_augmentation(cfg, False))

        names = ['object']
        DatasetCatalog.register(dataset_name, lambda :preprocess_txt2coco(cfg.DATASETS.TESTPATH))
        MetadataCatalog.get(dataset_name).set(thing_classes=names, evaluator_type='coco')

        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        output_folder = os.path.join(
            cfg.OUTPUT_DIR, "inference_{}".format(dataset_name))
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "lvis":
            evaluator = LVISEvaluator(dataset_name, cfg, True, output_folder)
        elif evaluator_type == 'coco':
            evaluator = COCOEvaluator(dataset_name, cfg, True, output_folder)
        else:
            assert 0, evaluator_type

        results[dataset_name] = inference_on_dataset(
            model, data_loader, evaluator)

        # if cfg.debug:
        #     predictions = evaluator._predictions

        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(
                dataset_name))
            print_csv_format(results[dataset_name])
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )

    start_iter = (
            checkpointer.resume_or_load(
                cfg.MODEL.WEIGHTS, resume=resume,
            ).get("iteration", -1) + 1
    )
    if cfg.SOLVER.RESET_ITER:
        logger.info('Reset loaded iteration. Start training from iteration 0.')
        start_iter = 0
    max_iter = cfg.SOLVER.MAX_ITER if cfg.SOLVER.TRAIN_ITER < 0 else cfg.SOLVER.TRAIN_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    mapper = DatasetMapper(cfg, True) if cfg.INPUT.CUSTOM_AUG == '' else \
        DatasetMapper(cfg, True, augmentations=build_custom_augmentation(cfg, True))

    # def read_json(path):
    #     with open(path, 'r') as f:
    #         dict = json.load(f)
    #     return dict

    names = ['object']
    # names = ['RBC', 'WBC', 'Platelets']
    DatasetCatalog.register(cfg.DATASETS.TRAIN[0], lambda :preprocess_txt2coco(cfg.DATASETS.TRAINPATH))
    MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).set(thing_classes=names, evaluator_type='coco')

    if cfg.DATALOADER.SAMPLER_TRAIN in ['TrainingSampler', 'RepeatFactorTrainingSampler']:
        data_loader = build_detection_train_loader(cfg, mapper=mapper)
    else:
        from centernet.data.custom_dataset_dataloader import build_custom_train_loader
        data_loader = build_custom_train_loader(cfg, mapper=mapper)

    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        step_timer = Timer()
        data_timer = Timer()
        start_time = time.perf_counter()
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            data_time = data_timer.seconds()
            storage.put_scalars(data_time=data_time)
            step_timer.reset()
            iteration = iteration + 1
            storage.step()
            loss_dict = model(data)

            losses = sum(
                loss for k, loss in loss_dict.items())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() \
                                 for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(
                    total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            storage.put_scalar(
                "lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

            step_time = step_timer.seconds()
            storage.put_scalars(time=step_time)
            data_timer.reset()
            scheduler.step()

            if (
                    cfg.TEST.EVAL_PERIOD > 0
                    and iteration % cfg.TEST.EVAL_PERIOD == 0
                    and iteration != max_iter
            ):
                do_test(cfg, model)
                comm.synchronize()

            if iteration - start_iter > 5 and \
                    (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

        total_time = time.perf_counter() - start_time
        logger.info(
            "Total training time: {}".format(
                str(datetime.timedelta(seconds=int(total_time)))))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_centernet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if '/auto' in cfg.OUTPUT_DIR:
        file_name = os.path.basename(args.config_file)[:-5]
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace('/auto', '/{}'.format(file_name))
        logger.info('OUTPUT_DIR: {}'.format(cfg.OUTPUT_DIR))
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        if cfg.TEST.AUG.ENABLED:
            logger.info("Running inference with test-time augmentation ...")
            model = GeneralizedRCNNWithTTA(cfg, model, batch_size=1)

        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
            find_unused_parameters=True
        )

    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model)


def custom_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
               or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file",
                        default="/home/centernet2/projects/CenterNet2/configs/CenterNet2_R2-101-DCN-BiFPN_4x+4x_1560_ST.yaml",
                        metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        default=True,
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
             "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", default=True, action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
             "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
             "See config references at "
             "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    args = custom_argument_parser()
    args.add_argument('--manual_device', default='')
    args = args.parse_args()
    if args.manual_device != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.manual_device
    args.dist_url = 'tcp://127.0.0.1:{}'.format(
        torch.randint(11111, 60000, (1,))[0].item())
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
