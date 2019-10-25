sample_submission_path = './input/severstal-steel-defect-detection/sample_submission.csv'
train_df_path = './input/severstal-steel-defect-detection/train.csv'
data_folder = "./input/severstal-steel-defect-detection/train_images"
test_data_folder = "./input/severstal-steel-defect-detection/test_images"
FOLDS_ids = './input/folds.pkl'
lb_test = './input/severstal-steel-defect-detection/submission_0.91625.csv'
isDebug = False
unet_encoder = 'se_resnext50_32x4d'
ATTENTION_TYPE = None  # None # Only for UNET scse
num_epochs = 100
LEARNING_RATE = 5e-4/100
BATCH_SIZE = {"train": 4, "val": 1}
TOTAL_FOLDS = 10
#model_weights = 'imagenet'
EARLY_STOPING = 30

model_weights = '/home/dex/Desktop/ml/Severstal-Steel-Defect-Detection/model_weights/fpn/model_se_resnext50_32x4d_fold_0_epoch_30_dice_0.9357 lb91/model_se_resnext50_32x4d_fold_0_epoch_7_dice_0.935771107673645.pth'
crop_image_size = None  # (256, 1600)
INITIAL_MINIMUM_DICE = 0.9

if isDebug:
    num_epochs = 1
