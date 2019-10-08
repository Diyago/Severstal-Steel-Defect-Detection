sample_submission_path = './input/severstal-steel-defect-detection/sample_submission.csv'
train_df_path = './input/severstal-steel-defect-detection/train.csv'
data_folder = "./input/severstal-steel-defect-detection/"
test_data_folder = "./input/severstal-steel-defect-detection/test_images"

isDebug = False
unet_encoder = 'resnet34'
attention_type = None  # scse
num_epochs = 20
LEARNING_RATE = 5e-4/2
batch_size = {"train": 8, "val": 6}
model_weights = [
    './model_weights/backup/model_resnet34_fold_0_epoch_6_dice_0.7099842373015202.pth',
    './model_weights/backup/model_resnet34_fold_1_epoch_15_dice_0.7354779279952208.pth',
    './model_weights/backup/model_resnet34_fold_2_epoch_18_dice_0.7640665367553647.pth',
    './model_weights/backup/model_resnet34_fold_3_epoch_18_dice_0.7427178866012297.pth',
    './model_weights/backup/model_resnet34_fold_4_epoch_19_dice_0.7650702263993822.pth'
]

crop_image_size = (256, 416)
INITIAL_MINIMUM_DICE = 0.65

if isDebug:
    num_epochs = 1
