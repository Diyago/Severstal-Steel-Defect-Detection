# Severstal-Steel-Defect-Detection
Can you detect and classify defects in steel? Segmentation in Pytorch

Project structure:
 * **common_blocks** - classes and functions for training
    - **dataloader.py** - dataloader of competition data    
    - **metric.py** - function with metric functions
    - **training_helper.py** - main class for training with cross validation or train_test_split
    - **utils.py** - useful utils for logs, model loading, mask transformation
    - **losses.py** - different segmentation losses
    - **optimizers.py** - SOTA optimizers
* **configs**
   - **train_params.py** (in development)
   1. **isDebug - then is True, pipelines finishes to work in minutes (useful for code testing)
   2. **unet_encoder** - you can specify unet encoder (resnet, resnext, se-resnext are available)
   3. **crop_image_size** - if specified will train on crops, othwerwise on full image size
   4. **attention_type** - will add scse blocks to decoder
   5. **path, folder** - paths to competition data
   
* **train.py** - main code for training
* **inference.py** (TODO - currently in kaggle kernels)

TODO
1. Try train single model on train