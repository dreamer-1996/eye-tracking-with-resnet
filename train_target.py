#!/usr/bin/env python

import time

import torch
import torchvision.utils
from fvcore.common.checkpoint import Checkpointer

from gaze_estimation import (
    create_dataloader,               #Present in gaze estimator/dataloader.py
    create_logger,                  #Present  in gaze estimator/logger.py
    create_loss,                    #Present in gaze estimator/losses.py
    create_optimizer,               #Present is gaze estimator/optim.py
    create_scheduler,               #Present in gaze estimator/scheduler.py
    create_tensorboard_writer,      #Present in gaze estimator/tensorboard.py
)
from gaze_estimation import GazeEstimationMethod, create_model   #Present in gaze estimator/types.py and gaze_estimation/models/_init.py respectively
from gaze_estimation.utils import (    #Present in gaze estimator/utils.py
    AverageMeter,
    compute_angle_error,
    compute_linear_error,
    create_train_output_dir,
    load_config,
    save_config,
    set_seeds,
    setup_cudnn,
)


def train(epoch, model, optimizer, scheduler, loss_function, train_loader,
          config, tensorboard_writer, logger):
    logger.info(f'Train {epoch}')

    model.train()

    device = torch.device(config.device)

    loss_meter = AverageMeter()  ########## Need to change this
    angle_error_meter = AverageMeter() ###### Need to change this

    start = time.time()
    for step, (images, poses, targets) in enumerate(train_loader):
        if config.tensorboard.train_images and step == 0:
            image = torchvision.utils.make_grid(images,
                                                normalize=True,
                                                scale_each=True)
            tensorboard_writer.add_image('Train/Image', image, epoch)

        images = images.to(device)
        poses = poses.to(device)
        #gazes = gazes.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()

        if config.mode == GazeEstimationMethod.MPIIGaze.name:
            outputs = model(images,poses)#model(images, poses)
        elif config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
            outputs = model(images)
        else:
            raise ValueError
        loss = loss_function(outputs,targets.float()) #loss_function(outputs, gazes)
        #print("Loss==>",loss)
        loss.backward()

        optimizer.step()

        #angle_error = compute_angle_error (outputs,targets).mean()       #compute_angle_error(outputs, gazes).mean()   #####Needs to be changed
        linear_error = compute_linear_error(outputs,targets.float()).mean()
        #print("Linear error===>",linear_error)
        num = images.size(0)
        loss_meter.update(loss.item(), num)
        angle_error_meter.update(linear_error.item(), num)

        if step % config.train.log_period == 0:
            logger.info(f'Epoch {epoch} '
                        f'Step {step}/{len(train_loader)} '
                        f'lr {scheduler.get_last_lr()[0]:.6f} '
                        f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        f'Linear error {angle_error_meter.val:.2f} '
                        f'({angle_error_meter.avg:.2f})')

    elapsed = time.time() - start
    logger.info(f'Elapsed {elapsed:.2f}')

    tensorboard_writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
    tensorboard_writer.add_scalar('Train/lr',
                                  scheduler.get_last_lr()[0], epoch)
    tensorboard_writer.add_scalar('Train/LinearError', angle_error_meter.avg,
                                  epoch)
    tensorboard_writer.add_scalar('Train/Time', elapsed, epoch)


def validate(epoch, model, loss_function, val_loader, config,
             tensorboard_writer, logger):
    logger.info(f'Val {epoch}')

    model.eval()

    device = torch.device(config.device)

    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()
    start = time.time()

    with torch.no_grad():
        for step, (images, poses, targets) in enumerate(val_loader):
            if config.tensorboard.val_images and epoch == 0 and step == 0:
                image = torchvision.utils.make_grid(images,
                                                    normalize=True,
                                                    scale_each=True)
                tensorboard_writer.add_image('Val/Image', image, epoch)

            images = images.to(device)
            poses = poses.to(device)
            #gazes = gazes.to(device)
            targets = targets.to(device)

            if config.mode == GazeEstimationMethod.MPIIGaze.name:
                outputs = model(images, poses)
            elif config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
                outputs = model(images)
            else:
                raise ValueError
            loss = loss_function(outputs, targets.float())

            #angle_error = compute_angle_error(outputs, gazes).mean()
            linear_error = compute_linear_error(outputs, targets.float()).mean()

            num = images.size(0)
            loss_meter.update(loss.item(), num)
            angle_error_meter.update(linear_error.item(), num)

    logger.info(f'Epoch {epoch} '
                f'loss {loss_meter.avg:.4f} '
                f'Linear error {angle_error_meter.avg:.2f}')

    elapsed = time.time() - start
    logger.info(f'Elapsed {elapsed:.2f}')

    if epoch > 0:
        tensorboard_writer.add_scalar('Val/Loss', loss_meter.avg, epoch)
        tensorboard_writer.add_scalar('Val/LinearError', angle_error_meter.avg,
                                      epoch)
    tensorboard_writer.add_scalar('Val/Time', elapsed, epoch)

    if config.tensorboard.model_params:
        for name, param in model.named_parameters():
            tensorboard_writer.add_histogram(name, param, epoch)


def main():
    config = load_config()           #load_confing present in util.py..reads from the config file the parameters of the model.here lenet_train.yaml is the config file

    set_seeds(config.train.seed)     #Utils.py......randomisation
    setup_cudnn(config)              #Utils.py......don't know the work

    output_dir = create_train_output_dir(config)     #utils.py....creates a directory "experiments/mpiigaze/lenet/exp00" as specified in config file
    save_config(config, output_dir)
    logger = create_logger(name=__name__,
                           output_dir=output_dir,
                           filename='log.txt')
    logger.info(config)

    train_loader, val_loader = create_dataloader(config, is_train=True) #function to load the data during training (training+validation) and testing

    model = create_model(config) #creating the model with the dataset name (mpiigaze/mpiifacegaze --available in gaze_estimation/models) and the model name like resent/lenet

    loss_function = create_loss(config)
    optimizer = create_optimizer(config, model)
    scheduler = create_scheduler(config, optimizer)
    checkpointer = Checkpointer(model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                save_dir=output_dir.as_posix(),
                                save_to_disk=True)
    tensorboard_writer = create_tensorboard_writer(config, output_dir)

    if config.train.val_first:
        validate(0, model, loss_function, val_loader, config,
                 tensorboard_writer, logger)

    for epoch in range(1, config.scheduler.epochs + 1):
        train(epoch, model, optimizer, scheduler, loss_function, train_loader,
              config, tensorboard_writer, logger)
        scheduler.step()

        if epoch % config.train.val_period == 0:
            validate(epoch, model, loss_function, val_loader, config,
                     tensorboard_writer, logger)

        if (epoch % config.train.checkpoint_period == 0
                or epoch == config.scheduler.epochs):
            checkpoint_config = {'epoch': epoch, 'config': config.as_dict()}
            checkpointer.save(f'checkpoint_{epoch:04d}', **checkpoint_config)

    tensorboard_writer.close()


if __name__ == '__main__':
    main()
