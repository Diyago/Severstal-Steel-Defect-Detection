import gc
from common_blocks.training_helper import Trainer_cv
from common_blocks.utils import plot, set_seed
from configs.train_params import num_epochs, model_weights, BATCH_SIZE
from common_blocks.utils import load_model_unet, load_model_fpn

if __name__ == '__main__':
    set_seed()
    for cur_fold in range(0, 5):
        print('Current FOLD {}'.format(cur_fold))
        model_trainer = Trainer_cv(load_model_unet(model_weights),
                                   num_epochs,
                                   cur_fold,
                                   batch_size=BATCH_SIZE)
        model_trainer.start()

        plot(model_trainer.losses, "BCE-DICE loss", cur_fold)
        plot(model_trainer.dice_scores, "Dice score", cur_fold)
        # plot(model_trainer.iou_scores, "IoU score", cur_fold)

        del model_trainer
        gc.collect()

