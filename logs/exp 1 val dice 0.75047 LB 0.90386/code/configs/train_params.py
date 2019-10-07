sample_submission_path = './input/severstal-steel-defect-detection/sample_submission.csv'
train_df_path = './input/severstal-steel-defect-detection/train.csv'
data_folder = "./input/severstal-steel-defect-detection/"
test_data_folder = "./input/severstal-steel-defect-detection/test_images"

isDebug = False
unet_encoder = 'resnet34'
attention_type = None  # scse
num_epochs = 20
model_weights = './model_weights/backup/model_fold_1_dice_0.5261443198132159.pth' #"imagenet",
crop_image_size = 256

if isDebug:
    num_epochs = 1
