# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pdb
import shutil
import time
import random

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather
from utils.data_utils import mask_rand_patch
from monai.data import decollate_batch



def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        data, target = data.cuda(args.rank), target.cuda(args.rank)

#   "MICCAI_BraTS_2018_Data_Training/HGG/Brats18_2013_10_1/Brats18_2013_10_1_flair.nii",
# "MICCAI_BraTS_2018_Data_Training/HGG/Brats18_2013_10_1/Brats18_2013_10_1_t1.nii",
# "MICCAI_BraTS_2018_Data_Training/HGG/Brats18_2013_10_1/Brats18_2013_10_1_t1ce.nii",
# "MICCAI_BraTS_2018_Data_Training/HGG/Brats18_2013_10_1/Brats18_2013_10_1_t2.nii"
        
        modalities_indices = [i for i, modality in enumerate(["Flair", "T1", "T1c", "T2"]) if modality in args.in_modalities]
        # modalities_indices = [random.randint(0,3)]
        data_s = data[:,modalities_indices,:,:,:]
        # data_s = data[:,[random.randint(0,3)],:,:,:]
        all_indices = np.arange(data.shape[1])
        exclude_indices = np.isin(all_indices, modalities_indices, invert=True)
        # random Mask:

        window_size = 16
        window_sizes = tuple(window_size for _ in range(3))
        input_sizes = (args.roi_x, args.roi_y, args.roi_z)
        data_s_masked, mask1 = mask_rand_patch(window_sizes, input_sizes, args.mask_ratio, data_s)

        # 使用布尔索引获取除了 modalities_indices 之外的数据
        data_except_modalities = data[:, exclude_indices, :, :, :]

        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            x_rec= model(data_s_masked)
            loss = loss_func(x_rec,data,mask1)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg




def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_sigmoid=None,
    post_pred=None,
    semantic_classes=None,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "rec_loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        # if (epoch + 1) % args.val_every == 0:
        #     if args.distributed:
        #         torch.distributed.barrier()
        #     epoch_time = time.time()
        #     val_acc = val_epoch(
        #         model,
        #         val_loader,
        #         epoch=epoch,
        #         acc_func=acc_func,
        #         model_inferer=model_inferer,
        #         args=args,
        #         post_sigmoid=post_sigmoid,
        #         post_pred=post_pred,
        #     )

        #     if args.rank == 0:
        #         Dice_TC = val_acc[0]
        #         Dice_WT = val_acc[1]
        #         Dice_ET = val_acc[2]
        #         print(
        #             "Final validation stats {}/{}".format(epoch, args.max_epochs - 1),
        #             ", Dice_TC:",
        #             Dice_TC,
        #             ", Dice_WT:",
        #             Dice_WT,
        #             ", Dice_ET:",
        #             Dice_ET,
        #             ", time {:.2f}s".format(time.time() - epoch_time),
        #         )

        #         if writer is not None:
        #             writer.add_scalar("Mean_Val_Dice", np.mean(val_acc), epoch)
        #             if semantic_classes is not None:
        #                 for val_channel_ind in range(len(semantic_classes)):
        #                     if val_channel_ind < val_acc.size:
        #                         writer.add_scalar(semantic_classes[val_channel_ind], val_acc[val_channel_ind], epoch)
        #         val_avg_acc = np.mean(val_acc)
        #         if val_avg_acc > val_acc_max:
        #             print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
        #             val_acc_max = val_avg_acc
        #             b_new_best = True
        #             if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
        #                 save_checkpoint(
        #                     model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
        #                 )

        if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
            save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
            # if b_new_best:
            #     print("Copying to model.pt new best model!!!!")
            #     shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

        # if args.rank == 0 and args.logdir is not None and args.save_checkpoint and epoch %10 == 0:
        #     save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename=f"model_{epoch}.pt")

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max
