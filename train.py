
import sys
import torch
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing
from multiprocessing import freeze_support
from datetime import datetime
import torchvision.transforms as T

import test
import util
import commons
import cosface_loss
import augmentations
from cosplace_model import cosplace_network
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset

import argparse


def parse_arguments(is_training: bool = True):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # CosPlace Groups parameters
    parser.add_argument("--M", type=int, default=5, help="_")
    parser.add_argument("--alpha", type=int, default=10, help="_")
    parser.add_argument("--N", type=int, default=5, help="_")
    parser.add_argument("--L", type=int, default=2, help="_")
    parser.add_argument("--groups_num", type=int, default=8, help="_")
    parser.add_argument("--min_images_per_class", type=int, default=1, help="_")
    # Model parameters
    parser.add_argument("--backbone", type=str, default="ResNet18",
                        choices=["VGG16",
                                 "ResNet18", "ResNet50", "ResNet101", "ResNet152",
                                 "EfficientNet_B0", "EfficientNet_B1", "EfficientNet_B2",
                                 "EfficientNet_B3", "EfficientNet_B4", "EfficientNet_B5",
                                 "EfficientNet_B6", "EfficientNet_B7"], help="_")
    parser.add_argument("--fc_output_dim", type=int, default=512,
                        help="Output dimension of final fully connected layer")
    parser.add_argument("--train_all_layers", default=False, action="store_true",
                        help="If true, train all layers of the backbone")
    # Training parameters
    parser.add_argument("--use_amp16", action="store_true",
                        help="use Automatic Mixed Precision")
    parser.add_argument("--augmentation_device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="on which device to run data augmentation")
    parser.add_argument("--batch_size", type=int, default=32, help="_")
    parser.add_argument("--epochs_num", type=int, default=50, help="_")
    parser.add_argument("--iterations_per_epoch", type=int, default=10000, help="_")
    parser.add_argument("--lr", type=float, default=0.00001, help="_")
    parser.add_argument("--classifiers_lr", type=float, default=0.01, help="_")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Width and height of training images (1:1 aspect ratio))")
    parser.add_argument("--resize_test_imgs", default=False, action="store_true",
                        help="If the test images should be resized to image_size along"
                             "the shorter side while maintaining aspect ratio")
    # Data augmentation
    parser.add_argument("--brightness", type=float, default=0.7, help="_")
    parser.add_argument("--contrast", type=float, default=0.7, help="_")
    parser.add_argument("--hue", type=float, default=0.5, help="_")
    parser.add_argument("--saturation", type=float, default=0.7, help="_")
    parser.add_argument("--random_resized_crop", type=float, default=0.5, help="_")
    # Validation / test parameters
    parser.add_argument("--infer_batch_size", type=int, default=16,
                        help="Batch size for inference (validating and testing)")
    parser.add_argument("--positive_dist_threshold", type=int, default=25,
                        help="distance in meters for a prediction to be considered a positive")
    # Resume parameters
    parser.add_argument("--resume_train", type=str, default=None,
                        help="path to checkpoint to resume, e.g. logs/.../last_checkpoint.pth")
    parser.add_argument("--resume_model", type=str, default=None,
                        help="path to model to resume, e.g. logs/.../best_model.pth")
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="_")
    parser.add_argument("--seed", type=int, default=0, help="_")
    parser.add_argument("--num_workers", type=int, default=8, help="_")
    parser.add_argument("--num_preds_to_save", type=int, default=0,
                        help="At the end of training, save N preds for each query. "
                             "Try with a small number like 3")
    parser.add_argument("--save_only_wrong_preds", action="store_true",
                        help="When saving preds (if num_preds_to_save != 0) save only "
                             "preds for difficult queries, i.e. with uncorrect first prediction")
    # Paths parameters
    if is_training:  # train and val sets are needed only for training
        parser.add_argument("--train_set_folder", type=str, required=True,
                            help="path of the folder with training images")
        parser.add_argument("--val_set_folder", type=str, required=True,
                            help="path of the folder with val images (split in database/queries)")
    parser.add_argument("--test_set_folder", type=str, required=True,
                        help="path of the folder with test images (split in database/queries)")
    parser.add_argument("--save_dir", type=str, default="default",
                        help="name of directory on which to save the logs, under logs/save_dir")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True  # Provides a speedup
    freeze_support()
    args = parse_arguments()
    start_time = datetime.now()
    args.output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
    commons.make_deterministic(args.seed)
    commons.setup_logging(args.output_folder, console="debug")
    logging.info(" ".join(sys.argv))
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.output_folder}")

    #### Model
    model = cosplace_network.GeoLocalizationNet(args.backbone, args.fc_output_dim, args.train_all_layers)

    logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

    if args.resume_model is not None:
        logging.debug(f"Loading model from {args.resume_model}")
        model_state_dict = torch.load(args.resume_model)
        model.load_state_dict(model_state_dict)

    model = model.to(args.device).train()

    #### Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    #### Datasets
    groups = [TrainDataset(args, args.train_set_folder, M=args.M, alpha=args.alpha, N=args.N, L=args.L,
                           current_group=n, min_images_per_class=args.min_images_per_class) for n in range(args.groups_num)]
    # Each group has its own classifier, which depends on the number of classes in the group
    classifiers = [cosface_loss.MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups]
    classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in classifiers]

    logging.info(f"Using {len(groups)} groups")
    logging.info(f"The {len(groups)} groups have respectively the following number of classes {[len(g) for g in groups]}")
    logging.info(f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")

    val_ds = TestDataset(args.val_set_folder, image_size=args.image_size, resize_test_imgs=args.resize_test_imgs)
    test_ds = TestDataset(args.test_set_folder, queries_folder="queries_v1", image_size=args.image_size, resize_test_imgs=args.resize_test_imgs)
    logging.info(f"Validation set: {val_ds}")
    logging.info(f"Test set: {test_ds}")

    #### Resume
    if args.resume_train:
        model, model_optimizer, classifiers, classifiers_optimizers, best_val_recall1, start_epoch_num = \
            util.resume_train(args, args.output_folder, model, model_optimizer, classifiers, classifiers_optimizers)
        model = model.to(args.device)
        epoch_num = start_epoch_num - 1
        logging.info(f"Resuming from epoch {start_epoch_num} with best R@1 {best_val_recall1:.1f} from checkpoint {args.resume_train}")
    else:
        best_val_recall1 = start_epoch_num = 0

    #### Train / evaluation loop
    logging.info("Start training ...")
    logging.info(f"There are {len(groups[0])} classes for the first group, " +
                 f"each epoch has {args.iterations_per_epoch} iterations " +
                 f"with batch_size {args.batch_size}, therefore the model sees each class (on average) " +
                 f"{args.iterations_per_epoch * args.batch_size / len(groups[0]):.1f} times per epoch")


    if args.augmentation_device == "cuda":
        gpu_augmentation = T.Compose([
                augmentations.DeviceAgnosticColorJitter(brightness=args.brightness,
                                                        contrast=args.contrast,
                                                        saturation=args.saturation,
                                                        hue=args.hue),
                augmentations.DeviceAgnosticRandomResizedCrop([args.image_size, args.image_size],
                                                              scale=[1-args.random_resized_crop, 1]),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    if args.use_amp16:
        scaler = torch.cuda.amp.GradScaler()

    for epoch_num in range(start_epoch_num, args.epochs_num):

        #### Train
        epoch_start_time = datetime.now()
        # Select classifier and dataloader according to epoch
        current_group_num = epoch_num % args.groups_num
        classifiers[current_group_num] = classifiers[current_group_num].to(args.device)
        util.move_to_device(classifiers_optimizers[current_group_num], args.device)

        dataloader = commons.InfiniteDataLoader(groups[current_group_num], num_workers=args.num_workers,
                                                batch_size=args.batch_size, shuffle=True,
                                                pin_memory=(args.device == "cuda"), drop_last=True)

        dataloader_iterator = iter(dataloader)
        model = model.train()

        epoch_losses = np.zeros((0, 1), dtype=np.float32)
        for iteration in tqdm(range(args.iterations_per_epoch), ncols=100):
            images, targets, _ = next(dataloader_iterator)
            images, targets = images.to(args.device), targets.to(args.device)

            if args.augmentation_device == "cuda":
                images = gpu_augmentation(images)

            model_optimizer.zero_grad()
            classifiers_optimizers[current_group_num].zero_grad()

            if not args.use_amp16:
                descriptors = model(images)
                output = classifiers[current_group_num](descriptors, targets)
                loss = criterion(output, targets)
                loss.backward()
                epoch_losses = np.append(epoch_losses, loss.item())
                del loss, output, images
                model_optimizer.step()
                classifiers_optimizers[current_group_num].step()
            else:  # Use AMP 16
                with torch.cuda.amp.autocast():
                    descriptors = model(images)
                    output = classifiers[current_group_num](descriptors, targets)
                    loss = criterion(output, targets)
                scaler.scale(loss).backward()
                epoch_losses = np.append(epoch_losses, loss.item())
                del loss, output, images
                scaler.step(model_optimizer)
                scaler.step(classifiers_optimizers[current_group_num])
                scaler.update()

        classifiers[current_group_num] = classifiers[current_group_num].cpu()
        util.move_to_device(classifiers_optimizers[current_group_num], "cpu")

        logging.debug(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                      f"loss = {epoch_losses.mean():.4f}")

        #### Evaluation
        recalls, recalls_str = test.test(args, val_ds, model)
        logging.info(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, {val_ds}: {recalls_str[:20]}")
        is_best = recalls[0] > best_val_recall1
        best_val_recall1 = max(recalls[0], best_val_recall1)
        # Save checkpoint, which contains all training parameters
        util.save_checkpoint({
            "epoch_num": epoch_num + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": model_optimizer.state_dict(),
            "classifiers_state_dict": [c.state_dict() for c in classifiers],
            "optimizers_state_dict": [c.state_dict() for c in classifiers_optimizers],
            "best_val_recall1": best_val_recall1
        }, is_best, args.output_folder)


    logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

    #### Test best model on test set v1
    best_model_state_dict = torch.load(f"{args.output_folder}/best_model.pth")
    model.load_state_dict(best_model_state_dict)

    logging.info(f"Now testing on the test set: {test_ds}")
    recalls, recalls_str = test.test(args, test_ds, model, args.num_preds_to_save)
    logging.info(f"{test_ds}: {recalls_str}")

    logging.info("Experiment finished (without any errors)")
