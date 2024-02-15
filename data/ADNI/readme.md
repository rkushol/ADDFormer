## Folder structure
Keep your 3D MRI data (.nii files) inside `train/test/valid` folders according to the following structure. 

Run `2d_slice_generator_third.py` to generate 2D files and store in `train_2D/test_2D/valid_2D` folders.

Run `data_list_generator_2D.py` and `data_list_generator_3D.py` to generate the list files.


data  
├── ADNI    
├  ├── AD_train   
├  ├── AD_test    
├  ├── AD_valid   
├  ├── CN_train   
├  ├── CN_test   
├  ├── CN_valid   
├  ├── AD_train_2D   
├  ├── AD_test_2D    
├  ├── AD_valid_2D   
├  ├── CN_train_2D   
├  ├── CN_test_2D   
├  ├── CN_valid_2D   

