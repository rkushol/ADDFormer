import nibabel as nib
import os
import glob
import numpy as np


if __name__ == "__main__":

    dataset_names= ["AD_valid","CN_valid","AD_test","CN_test","AD_train","CN_train"]

    for dataset_name in dataset_names:
        print(dataset_name)
        for entry in os.listdir("."):
            if os.path.isdir(entry) and (entry == dataset_name):
                full_path = glob.glob(os.path.join(entry, "**/*.nii"), recursive=True)            
                for nii_file in full_path:
                    filename = os.path.split(nii_file)[1]
                    #print(nii_file)
                    print(filename)
                    data = nib.load(nii_file)
                    fdata = data.get_fdata()               
                    
                    for slice_number in range(112,127):                   
                        new_filename= "third_"+str(slice_number)+"_"+filename
                        #print(new_filename)
                        fdata_2D = fdata[:, :, slice_number].astype(np.float32)
                        #print("fdata_2D: ", fdata_2D.shape)
                        #print("fdata_2D: ", fdata_2D.dtype)

                        save_path = os.path.join( dataset_name +"_2D", new_filename)
                        nib.save(nib.Nifti1Image(fdata_2D, affine = np.eye(4)), save_path)
                        #nib.save(nib.Nifti1Image(fdata_2D, None, header=data.header.copy()), save_path)