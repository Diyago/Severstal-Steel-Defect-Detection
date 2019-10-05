import gc
from common_blocks.dataloader import Trainer_cv
from segmentation_models_pytorch import Unet
from common_blocks.utils import plot, set_seed
from configs.train_params import unet_encoder, attention_type

if __name__ == '__main__':
    set_seed()
    model = Unet(unet_encoder, encoder_weights="imagenet", classes=4, activation=None,
                 attention_type=attention_type)
    for cur_fold in range(0, 5):
        print('Current FOLD {}'.format(cur_fold))
        model_trainer = Trainer_cv(model, cur_fold)
        model_trainer.start()
        # PLOT TRAINING
        losses = model_trainer.losses
        dice_scores = model_trainer.dice_scores  # overall dice
        iou_scores = model_trainer.iou_scores
        plot(losses, "BCE loss", cur_fold)
        plot(dice_scores, "Dice score", cur_fold)
        plot(iou_scores, "IoU score", cur_fold)

        del model_trainer
        gc.collect()
