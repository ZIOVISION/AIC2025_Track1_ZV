# Copyright (c) V-DETR authors. All Rights Reserved.
import argparse
import os
import sys
import pickle

import numpy as np
import torch
from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader, DistributedSampler
import MinkowskiEngine as ME

from datasets import build_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from optimizer import build_optimizer
from criterion import build_criterion
from utils.dist import init_distributed, is_distributed, is_primary, get_rank, barrier
from utils.misc import my_worker_init_fn
from utils.io import save_checkpoint, resume_if_possible

import wandb

import yaml
from types import SimpleNamespace

def wandb_log(*args, **kwargs):
    if is_primary():
        wandb.log(*args, **kwargs)
        


def load_config():
    parser = argparse.ArgumentParser(description="V-DETR config loader")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--ngpus', type=int, default=1, help='Number of GPUs to use')

    cli_args = parser.parse_args()

    # Load YAML config
    with open(cli_args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    config_dict['ngpus'] = cli_args.ngpus

    return SimpleNamespace(**config_dict)


def auto_reload(args):
    ignore_keys = [
    "test_only", "auto_test", "test_no_nms", "no_3d_nms", "rotated_nms",
    "nms_iou", "empty_pt_thre", "conf_thresh", "test_ckpt", "angle_nms",
    "angle_conf", "use_old_type_nms", "no_cls_nms", "filt_empty",
    "no_per_class_proposal", "use_cls_confidence_only", "test_size",
    "ngpus","dist_url","model_name","dataset_root_dir","meta_data_dir","checkpoint_dir",
    ]

    ckpt = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
    ckpt_args = ckpt["args"]
    for arg_name in vars(ckpt_args):
        if arg_name not in ignore_keys and hasattr(args, arg_name):
            if getattr(args, arg_name) != getattr(ckpt_args, arg_name):
                print(arg_name,getattr(args, arg_name),getattr(ckpt_args, arg_name))
                setattr(args, arg_name, getattr(ckpt_args, arg_name))



def do_train(
    args,
    model,
    model_no_ddp,
    optimizer,
    criterion,
    dataset_config,
    dataloaders,
    best_val_metrics,
):
    """
    Main training loop.
    This trains the model for `args.max_epoch` epochs and tests the model after every `args.eval_every_epoch`.
    We always evaluate the final checkpoint and report both the final AP and best AP on the val set.
    """

    num_iters_per_epoch = len(dataloaders["train"])
    num_iters_per_eval_epoch = len(dataloaders["test"])
    print(f"Model is {model}")
    print(f"Training started at epoch {args.start_epoch} until {args.max_epoch}.")
    print(f"One training epoch = {num_iters_per_epoch} iters.")
    print(f"One eval epoch = {num_iters_per_eval_epoch} iters.")

    final_eval = os.path.join(args.checkpoint_dir, "final_eval.txt")
    final_eval_pkl = os.path.join(args.checkpoint_dir, "final_eval.pkl")

    if os.path.isfile(final_eval):
        print(f"Found final eval file {final_eval}. Skipping training.")
        return

    for epoch in range(args.start_epoch, args.max_epoch):
        if is_distributed():
            dataloaders["train_sampler"].set_epoch(epoch)
            
        aps, wandb_iter, wandb_lr, wandb_loss, wandb_loss_details = train_one_epoch(
            args,
            epoch,
            model,
            optimizer,
            criterion,
            dataset_config,
            dataloaders["train"],
        )

        # latest checkpoint is always stored in checkpoint.pth
        save_checkpoint(
            args.checkpoint_dir,
            model_no_ddp,
            optimizer,
            epoch,
            args,
            best_val_metrics,
            filename="checkpoint.pth",
        )

        curr_iter = epoch * len(dataloaders["train"])
        use_evaluate = ((epoch != 0) and (epoch % args.eval_every_epoch == 0 or epoch == (args.max_epoch - 1))) or (epoch == 10)

        if is_primary():
            log_message = dict(\
                lr=wandb_lr,
                loss=wandb_loss,
                loss_cls=wandb_loss_details['loss_sem_cls'].item(),
                loss_angle_cls=wandb_loss_details['loss_angle_cls'].item(),
                loss_angle_reg=wandb_loss_details['loss_angle_reg'].item(),
                loss_center=wandb_loss_details['loss_center'].item() if 'loss_center' in wandb_loss_details else 0.0,
                loss_size=wandb_loss_details['loss_size'].item() if 'loss_size' in wandb_loss_details else 0.0,
                loss_giou=wandb_loss_details['loss_giou'].item() if 'loss_giou' in wandb_loss_details else 0.0,
                )
            if 'enc_point_cls_loss' in wandb_loss_details:
                log_message_enc=dict(\
                    enc_point_cls_loss=wandb_loss_details['enc_point_cls_loss'].item(),
                    )
                log_message.update(log_message_enc)#enc_point_cls_loss

            if args.wandb_activate:
                wandb_log(
                    data=log_message,
                    step=wandb_iter,
                    commit=not use_evaluate
                )

        # if (
        #     epoch > args.max_epoch * 0.90
        #     and args.save_separate_checkpoint_every_epoch > 0
        #     and epoch % args.save_separate_checkpoint_every_epoch == 0
        # ) or (epoch > args.max_epoch * 0.6 and epoch % args.eval_every_epoch == 0):
        #     # separate checkpoints are stored as checkpoint_{epoch}.pth
        save_checkpoint(
            args.checkpoint_dir,
            model_no_ddp,
            optimizer,
            epoch,
            args,
            best_val_metrics,
        )

        if use_evaluate:
            if epoch > args.max_epoch * 0.6:
                save_checkpoint(
                    args.checkpoint_dir,
                    model_no_ddp,
                    optimizer,
                    epoch,
                    args,
                    best_val_metrics,
                )
            ap_calculator, wandb_val_loss, wandb_val_loss_details = evaluate(
                args,
                epoch,
                model,
                criterion,
                dataset_config,
                dataloaders["test"],
                curr_iter,
            )
            metrics = ap_calculator.compute_metrics()
            ap25 = metrics[0.25]["mAP"]
            metric_str = ap_calculator.metrics_to_str(metrics, per_class=True)
            metrics_dict = ap_calculator.metrics_to_dict(metrics)
            if is_primary():
                print("==" * 10)
                print(f"Evaluate Epoch [{epoch}/{args.max_epoch}]; Metrics {metric_str}")
                print("==" * 10)

                log_message = dict(\
                    val_loss=wandb_val_loss,
                    val_AP25=metrics_dict['mAP_0.25'],
                    val_AP50=metrics_dict['mAP_0.5'],
                    val_AR25=metrics_dict['AR_0.25'],
                    val_AR50=metrics_dict['AR_0.5'],
                    val_loss_cls=wandb_val_loss_details['loss_sem_cls'].item(),
                    val_loss_angle_cls=wandb_val_loss_details['loss_angle_cls'].item(),
                    val_loss_angle_reg=wandb_val_loss_details['loss_angle_reg'].item(),
                    val_loss_center=wandb_val_loss_details['loss_center'].item(),
                    val_loss_size=wandb_val_loss_details['loss_size'].item(),               
                    val_loss_giou=wandb_val_loss_details['loss_giou'].item() if 'loss_giou' in wandb_val_loss_details else 0.0,) 
                if 'enc_point_cls_loss' in wandb_val_loss_details:
                    log_message_enc=dict(\
                        val_enc_point_cls_loss=wandb_val_loss_details['enc_point_cls_loss'].item(),
                        )
                    log_message.update(log_message_enc)
                if args.wandb_activate:
                    wandb_log(
                        data=log_message,
                        step=wandb_iter,
                    )
                
            if is_primary() and (
                len(best_val_metrics) == 0 or best_val_metrics[0.25]["mAP"] < ap25
            ):
                best_val_metrics = metrics
                filename = "checkpoint_best.pth"
                save_checkpoint(
                    args.checkpoint_dir,
                    model_no_ddp,
                    optimizer,
                    epoch,
                    args,
                    best_val_metrics,
                    filename=filename,
                )
                print(
                    f"Epoch [{epoch}/{args.max_epoch}] saved current best val checkpoint at {filename}; ap25 {ap25}"
                )

    # always evaluate last checkpoint
    epoch = args.max_epoch - 1
    curr_iter = epoch * len(dataloaders["train"])
    ap_calculator, wandb_val_loss, wandb_val_loss_details = evaluate(
        args,
        epoch,
        model,
        criterion,
        dataset_config,
        dataloaders["test"],
        curr_iter,
    )
    metrics = ap_calculator.compute_metrics()
    metric_str = ap_calculator.metrics_to_str(metrics)
    if is_primary():
        print("==" * 10)
        print(f"Evaluate Final [{epoch}/{args.max_epoch}]; Metrics {metric_str}")
        print("==" * 10)

        with open(final_eval, "w") as fh:
            fh.write("Training Finished.\n")
            fh.write("==" * 10)
            fh.write("Final Eval Numbers.\n")
            fh.write(metric_str)
            fh.write("\n")
            fh.write("==" * 10)
            fh.write("Best Eval Numbers.\n")
            fh.write(ap_calculator.metrics_to_str(best_val_metrics))
            fh.write("\n")

        with open(final_eval_pkl, "wb") as fh:
            pickle.dump(metrics, fh)


def test_model(args, model, model_no_ddp, criterion, dataset_config, dataloaders):
    if args.test_ckpt is None or not os.path.isfile(args.test_ckpt):
        f"Please specify a test checkpoint using --test_ckpt. Found invalid value {args.test_ckpt}"
        sys.exit(1)

    sd = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
    model_no_ddp.load_state_dict(sd["model"],strict=False)
    criterion = None  # do not compute loss for speed-up; Comment out to see test loss
    epoch = -1
    curr_iter = 0
    ap_calculator, _ , _ = evaluate(
        args,
        epoch,
        model,
        criterion,
        dataset_config,
        dataloaders["test"],
        curr_iter,
    )
    metrics = ap_calculator.compute_metrics()
    metric_str = ap_calculator.metrics_to_str(metrics)
    if is_primary():
        print("==" * 10)
        print(f"Test model; Metrics {metric_str}")
        print("==" * 10)
    if args.test_size:
        metrics = ap_calculator.compute_metrics(size='S')
        metric_str = ap_calculator.metrics_to_str(metrics)
        if is_primary():
            print("==" * 10)
            print(f"Test model S; Metrics {metric_str}")
            print("==" * 10)
        metrics = ap_calculator.compute_metrics(size='M')
        metric_str = ap_calculator.metrics_to_str(metrics)
        if is_primary():
            print("==" * 10)
            print(f"Test model M; Metrics {metric_str}")
            print("==" * 10)
        metrics = ap_calculator.compute_metrics(size='L')
        metric_str = ap_calculator.metrics_to_str(metrics)
        if is_primary():
            print("==" * 10)
            print(f"Test model L; Metrics {metric_str}")
            print("==" * 10)


def main(local_rank, args):
    if args.ngpus > 1:
        print(
            "Initializing Distributed Training. This is in BETA mode and hasn't been tested thoroughly. Use at your own risk :)"
        )
        print("To get the maximum speed-up consider reducing evaluations on val set by setting --eval_every_epoch to greater than 50")
        init_distributed(
            local_rank,
            global_rank=local_rank,
            world_size=args.ngpus,
            dist_url=args.dist_url,
            dist_backend="nccl",
        )

    print(f"Called with args: {args}")
    torch.cuda.set_device(local_rank)
    np.random.seed(args.seed + get_rank())
    torch.manual_seed(args.seed + get_rank())
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + get_rank())
    if args.test_only and args.auto_test:
        auto_reload(args)

    datasets, dataset_config = build_dataset(args)
    model = build_model(args, dataset_config)
    if args.test_ckpt != "":
        print(f"Loading checkpoint from {args.test_ckpt}...")
        sd = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
        checkpoint_state_dict = sd["model"]
        model_state_dict = model.state_dict() 

        filtered_state_dict = {}
        for k, v in checkpoint_state_dict.items():
            if k in model_state_dict and v.shape == model_state_dict[k].shape:
                filtered_state_dict[k] = v
            else:
                print(f"Skipping layer {k} due to shape mismatch.")

        # 필터링된 state_dict를 로드
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)

        print("Successfully loaded pre-trained weights.")
        if missing_keys:
            print("Missing keys (correctly ignored angle heads):", missing_keys)
        if unexpected_keys:
            print("Unexpected keys:", unexpected_keys)


    # # 1. 모델의 모든 파라미터를 우선 동결(freeze)시킵니다.
    # print("Freezing all model parameters...")
    # for param in model.parameters():
    #     param.requires_grad = False

    # # 2. angle_residual_head 부분의 파라미터만 학습하도록 활성화(unfreeze)합니다.
    # # vdetr_transformer.py를 보면 mlp_heads는 ModuleList 또는 ModuleDict일 수 있습니다.
    # print("Unfreezing angle_residual_head parameters...")
    # if isinstance(model.decoder.mlp_heads, torch.nn.ModuleList):
    #     # mlp_sep=True인 경우, ModuleList 안에 여러 ModuleDict가 존재
    #     for layer_heads in model.decoder.mlp_heads:
    #         if "angle_residual_head" in layer_heads:
    #             for param in layer_heads["angle_residual_head"].parameters():
    #                 param.requires_grad = True
    # elif isinstance(model.decoder.mlp_heads, torch.nn.ModuleDict):
    #     # mlp_sep=False인 경우, 단일 ModuleDict
    #     if "angle_residual_head" in model.decoder.mlp_heads:
    #         for param in model.decoder.mlp_heads["angle_residual_head"].parameters():
    #             param.requires_grad = True
    
    # # 학습될 파라미터 이름과 개수를 확인합니다.
    # print("\n========== Trainable Parameters ==========")
    # trainable_params_count = 0
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    #         trainable_params_count += param.numel()
    # print(f"Total trainable parameters: {trainable_params_count}")
    # print("========================================\n")

    model = model.cuda(local_rank)
    model_no_ddp = model

    if is_distributed():
        if args.mink_syncbn:
            model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank],find_unused_parameters = True,
        )
    criterion = build_criterion(args, dataset_config)
    criterion = criterion.cuda(local_rank)

    dataloaders = {}
    if args.test_only:
        dataset_splits = ["test"]
    else:
        dataset_splits = ["train", "test"]
    for split in dataset_splits:
        if split == "train":
            shuffle = True
        else:
            shuffle = False
        if is_distributed():
            sampler = DistributedSampler(datasets[split], shuffle=shuffle)
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(datasets[split])
        else:
            sampler = torch.utils.data.SequentialSampler(datasets[split])
        
        dataloaders[split] = DataLoader(
            datasets[split],
            sampler=sampler,
            batch_size=args.batchsize_per_gpu,
            num_workers=args.dataset_num_workers,
            worker_init_fn=my_worker_init_fn,
            collate_fn=datasets[split].collate_fn
        )
        dataloaders[split + "_sampler"] = sampler

    if args.test_only:
        criterion = None  # faster evaluation
        test_model(args, model, model_no_ddp, criterion, dataset_config, dataloaders)
    else:
        assert (
            args.checkpoint_dir is not None
        ), f"Please specify a checkpoint dir using --checkpoint_dir"
        if is_primary() and not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir, exist_ok=True)
        if is_primary():
            # setup wandb
            if args.wandb_activate:
                wandb.login(key=args.wandb_key)
                run = wandb.init(
                    id=args.checkpoint_dir.split('/')[-1],
                    name=args.checkpoint_dir.split('/')[-1],
                    entity=args.wandb_entity,
                    project=args.wandb_project,
                    config=args,
                )



        optimizer = build_optimizer(args, model_no_ddp)
        loaded_epoch, best_val_metrics = resume_if_possible(
            args.checkpoint_dir, model_no_ddp, optimizer
        )
        args.start_epoch = loaded_epoch + 1
        do_train(
            args,
            model,
            model_no_ddp,
            optimizer,
            criterion,
            dataset_config,
            dataloaders,
            best_val_metrics,
        )


def launch_distributed(args):
    world_size = args.ngpus
    if world_size == 1:
        main(local_rank=0, args=args)
    else:
        torch.multiprocessing.spawn(main, nprocs=world_size, args=(args,))


if __name__ == "__main__":
    #parser = make_args_parser()
    args = load_config()
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    launch_distributed(args)
