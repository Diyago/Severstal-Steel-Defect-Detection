sample_submission_path = './input/severstal-steel-defect-detection/sample_submission.csv'
train_df_path = './input/severstal-steel-defect-detection/train.csv'
data_folder = "./input/severstal-steel-defect-detection/"
test_data_folder = "./input/severstal-steel-defect-detection/test_images"
FOLDS_ids = './input/folds.pkl'

isDebug = False
unet_encoder = 'se_resnext50_32x4d'
ATTENTION_TYPE = None  # None # Only for UNET scse
num_epochs = 45
LEARNING_RATE = 5e-4
BATCH_SIZE = {"train": 4, "val": 1}
TOTAL_FOLDS = 10
model_weights = 'imagenet'
EARLY_STOPING = 15

#model_weights = './model_weights/fpn/model_se_resnext50_32x4d_fold_0_epoch_26_dice_0.94392.pth'
crop_image_size =  None#(256, 1600)
INITIAL_MINIMUM_DICE = 0.9

if isDebug:
    num_epochs = 1
