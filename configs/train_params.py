sample_submission_path = './input/severstal-steel-defect-detection/sample_submission.csv'
train_df_path = './input/severstal-steel-defect-detection/train.csv'
data_folder = "./input/severstal-steel-defect-detection/"
test_data_folder = "./input/severstal-steel-defect-detection/test_images"
FOLDS_ids = './input/folds.pkl'

isDebug = False
unet_encoder = 'inceptionresnetv2'
ATTENTION_TYPE = None  # None # Only for UNET scse
num_epochs = 45
LEARNING_RATE = 5e-4
BATCH_SIZE = {"train": 3, "val": 1}
TOTAL_FOLDS = 10
model_weights = 'imagenet'
EARLY_STOPING = 15

#model_weights = './model_weights/model_se_resnext101_32x4d_fold_0_epoch_38_dice_0.9300873875617981.pth'
crop_image_size = (256, 1600)
INITIAL_MINIMUM_DICE = 0.9

if isDebug:
    num_epochs = 1
