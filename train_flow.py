import os
import argparse
# from accelerate import Accelerator
from torch.optim import lr_scheduler
from torchfitter.trainer import Trainer
from torchfitter.callbacks import (
    LearningRateScheduler,
    RichProgressBar,
    LoggerCallback,
    # EarlyStopping
)
import models
import utils

__name2metric__ = {"multi_label": "MultiLabelAccuracyMetrics", "multi_class":"MultiClassAccuracyMetrics"}
__name2loss__ = {"multi_label": "WeatherDetMultiLabelLoss", "multi_class":"WeatherDetMultiClassLoss"}

def parse_args():
    parser = argparse.ArgumentParser()
    #model details
    parser.add_argument("consensus", default="conv3d", type=str, choices=["conv3d", "crossattn"])
    parser.add_argument("head", default="mlh", type=str, choices=["mlh", "mcsh", "mcmh", "mcmha"])
    parser.add_argument("--num_labels", default=7, type=int)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--is_2d", action='store_true')

    #dataloader details
    parser.add_argument("--data_root", default="/mnt", type=str)
    parser.add_argument("--data_type", default="multi_label", type=str, choices=["multi_label", "multi_class"], help="type of dataset")
    parser.add_argument("--bs", default=16, type=int, help="batch size")
    parser.add_argument("--nw", default=16, type=int, help="number of workers")
    parser.add_argument("--format", default="CTHW", type=str, choices=["TCHW", "CTHW"], help="format of input data")
    parser.add_argument("--data_aug", default="default", type=str, choices=["randaug", "default"], help="data augmentation")

    #loss, metrics details
    parser.add_argument("--metric_th", default=0.5, type=float)
    parser.add_argument("--metric_ct", default="hamming", type=str, choices=["exact_match", "hamming"])
    
    #optimizer details
    parser.add_argument("--optimizer", default="SGD", type=str, choices=['SGD', 'Adam', 'AdamW', 'RMSProp'])
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=5e-4, type=float)

    #training details
    parser.add_argument("--epochs", default=60, type=int, help="number of epochs")

    #logging details
    parser.add_argument("--project", default='runs/default_aug/', type=str, help="Folder to save logs and checkpoints")
    return parser.parse_args()

def main():
    args = parse_args()
    print(args.data_aug)
    name = f'tsm_{args.consensus}_{args.head}_{args.optimizer}'

    model = models.MMRecognizer3D(consensus=args.consensus, 
                                  cls_head=args.head, 
                                  num_labels=args.num_labels, 
                                  dropout_rate=args.dropout, 
                                  is_3d=not args.is_2d)

    criterion = getattr(utils, __name2loss__[args.data_type])()
    metrics = [getattr(utils, __name2metric__[args.data_type])(criteria=args.metric_ct, threshold=args.metric_th)]
    optimizer = utils.smart_optimizer(model, name=args.optimizer, lr=args.lr, momentum=args.momentum, decay=args.weight_decay)

    #dataloader
    train_loader = utils.create_video_opt_dataloader(root_path=args.data_root, 
                                                    data_list=args.data_type+'_train.txt', 
                                                    batch_size=args.bs,
                                                    is_train=True, 
                                                    num_worker=args.nw,
                                                    aug_type=args.data_aug,
                                                    format=args.format)
    val_loader = utils.create_video_opt_dataloader(root_path=args.data_root, 
                                                data_list=args.data_type+'_test.txt', 
                                                batch_size=args.bs,
                                                is_train=False, 
                                                num_worker=args.nw,
                                                format=args.format)

    ## Callbacks
    callbacks = []
    #scheduler 
    sch = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=6)
    callbacks.append(LearningRateScheduler(scheduler=sch, metric=__name2metric__[args.data_type])) #val on train

    #logger LoggerCallback
    # callbacks.append(RichProgressBar(precision=3))
    callbacks.append(LoggerCallback(update_step=100, precision=3))
    callbacks.append(utils.TensorBoardLogger(log_dir=os.path.join(args.project, name), 
                                             metric=__name2metric__[args.data_type]))

    #checkpointing
    callbacks.append(utils.ModelCheckpoint(ckpt_path=os.path.join(args.project, name, 'ckpt'), 
                                            monitor=__name2metric__[args.data_type],
                                            save_best=True, 
                                            save_optimizer=True, 
                                            mode='max', 
                                            save_freq=5))
    
    # accelerator = Accelerator(
    #             mixed_precision='no',
    #             gradient_accumulation_steps=1,
    #             step_scheduler_with_optimizer=True,
    #             project_dir=os.path.join(args.project, name),
    #             # log_with=["tensorboard"],
    #         )
    
    trainer = Trainer(
        model=model,  
        criterion=criterion,
        optimizer=optimizer,
        mixed_precision="no",
        metrics = metrics,
        # accelerator = accelerator,
        accumulate_iter=1, # accumulate gradient every 4 iterations
        gradient_clipping='norm',
        gradient_clipping_kwargs={'max_norm': 5.0, 'norm_type': 2.0},
        callbacks=callbacks #, early_stopping]
    )
    history = trainer.fit(train_loader, val_loader, epochs=args.epochs)

if __name__=='__main__':
    main()


#### Extras
# sch = lr_scheduler.MultiStepLR(optimizer, milestones=[40,45], gamma=0.1)
# early_stopping = EarlyStopping(patience=10, load_best=True, path=os.path.join(args.ckpt_path, "early_ckpt.pth"))
# logger = RichProgressBar(precision=3) #LoggerCallback(update_step=50)