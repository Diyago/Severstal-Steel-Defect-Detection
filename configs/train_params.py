sample_submission_path = './input/severstal-steel-defect-detection/sample_submission.csv'
train_df_path = './input/severstal-steel-defect-detection/train.csv'
data_folder = "./input/severstal-steel-defect-detection/"
test_data_folder = "./input/severstal-steel-defect-detection/test_images"

isDebug = False
unet_encoder = 'resnet34'
ATTENTION_TYPE = None  # None # Only for UNET scse
num_epochs = 30
LEARNING_RATE = 5e-4
BATCH_SIZE = {"train": 20, "val": 3}
model_weights = 'imagenet'
EARLY_STOPING = 10
# [
#     './model_weights/backup/model_resnet34_fold_0_epoch_6_dice_0.7099842373015202.pth',
#     './model_weights/backup/model_resnet34_fold_1_epoch_15_dice_0.7354779279952208.pth',
#     './model_weights/backup/model_resnet34_fold_2_epoch_18_dice_0.7640665367553647.pth',
#     './model_weights/backup/model_resnet34_fold_3_epoch_18_dice_0.7427178866012297.pth',
#     './model_weights/backup/model_resnet34_fold_4_epoch_19_dice_0.7650702263993822.pth'
# ]

crop_image_size = (256, 416)
INITIAL_MINIMUM_DICE = 0.85

if isDebug:
    num_epochs = 1
