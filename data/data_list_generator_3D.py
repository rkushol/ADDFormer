import os
import glob
import random

if __name__ == "__main__":
    #train_file = open("train_ADNI_2D.txt", "w")
    test_file = open("test_ADNI_3D.txt", "w")
    valid_file = open("valid_ADNI_3D.txt", "w")

    #train_list = []
    test_list = []
    valid_list = []
    data_dir = "."
    
    for entry in os.listdir("."):
        #entry_full_path = os.path.join(data_dir, entry)
        if os.path.isdir(entry):
            nii_files = glob.glob(os.path.join(entry, "**/*.nii"), recursive=True)
            #print(len(nii_files))
            category = 0
            nii_filtered = []
            for nii_file in nii_files:
                if "test_2D" not in nii_file:
                    if "valid_2D" not in nii_file:
                        nii_filtered.append(nii_file)

        	
            nii_files = nii_filtered
            for nii_file in nii_files:
                category = 0 if "CN_" in nii_file else (1 if "AD_" in nii_file else (2 if "MCI_" in nii_file else -1))
                if "test" in nii_file:
                    test_list.append((nii_file, category))
                elif "valid" in nii_file:
                    valid_list.append((nii_file, category))
                #elif "train_2D" in nii_file:
                #    train_list.append((nii_file, category))

    #random.shuffle(train_list)
    random.shuffle(valid_list)
    random.shuffle(test_list)

    #for data in train_list:
    #    train_file.write("{} {}\n".format(data[0], data[1]))

    for data in test_list:
        test_file.write("{} {}\n".format(data[0], data[1]))
        
    for data in valid_list:
        valid_file.write("{} {}\n".format(data[0], data[1]))

    #train_file.close()
    test_file.close()
    valid_file.close()
