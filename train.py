import gc
from common_blocks.dataloader import Trainer_cv
from segmentation_models_pytorch import Unet
from common_blocks.utils import plot, set_seed

if __name__ == '__main__':
    set_seed()
    model = Unet("resnet18", encoder_weights="imagenet", classes=4, activation=None)
    for cur_fold in range(0, 5):
        print('Current FOLD {}'.format(cur_fold))
        model_trainer = Trainer_cv(model, cur_fold)
        model_trainer.start()
        # PLOT TRAINING
        losses = model_trainer.losses
        dice_scores = model_trainer.dice_scores  # overall dice
        iou_scores = model_trainer.iou_scores
        plot(losses, "BCE loss")
        plot(dice_scores, "Dice score")
        plot(iou_scores, "IoU score")

        del model_trainer
        gc.collect()
