# ADDFormer
Alzheimer's Disease Detection from structural MRI using Fusion Transformer.

The paper has been published in the Conference of ISBI 2022.
Link: [https://doi.org/10.3390/diagnostics13182947](https://ieeexplore.ieee.org/iel7/9761376/9761399/09761421.pdf).

```
@inproceedings{kushol2022addformer,
  title={Addformer: Alzheimer’s disease detection from structural mri using fusion transformer},
  author={Kushol, Rafsanjany and Masoumzadeh, Abbas and Huo, Dong and Kalra, Sanjay and Yang, Yee-Hong},
  booktitle={2022 IEEE 19th International Symposium On Biomedical Imaging (ISBI)},
  pages={1--5},
  year={2022},
  organization={IEEE}
}
```

## Abstract
Alzheimer's disease is the most prevalent neurodegenerative disorder characterized by degeneration of the brain. It is classified as a brain disease causing dementia that presents with memory loss and cognitive impairment. Experts primarily use brain imaging and other tests to rule out the disease. To automatically detect Alzheimer's patients from healthy controls, this study adopts the vision transformer architecture, which can effectively capture the global or long-range relationship of image features. To further enhance the network's performance, frequency and image domain features are fused together since MRI data is acquired in the frequency domain before being transformed to images. We train the model with selected coronal 2D slices to leverage the transfer learning property of pre-training the network using ImageNet. Finally, the majority voting of the coronal slices of an individual subject is used to generate the final classification score. Our proposed method has been evaluated on the publicly available benchmark dataset ADNI. The experimental results demonstrate the advantage of our proposed approach in terms of classification accuracy compared with that of the state-of-the-art methods.


## Requirements
PyTorch  
nibabel  
scipy  
scikit-image  


## Datasets
ADNI dataset can be downloaded from [ADNI](http://adni.loni.usc.edu/) (Alzheimer’s Disease Neuroimaging Initiative)


## Preprocessing
### Skull stripping using Freesurfer v7.3.2
Command ``mri_synthstrip -i input -o stripped``

Details can be found [SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) (SynthStrip: Skull-Stripping for Any Brain Image)


### Registration to MNI-152 using FSL FLIRT function
Details can be found [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT)

One implementation can be found [here](https://github.com/vkola-lab/brain2020/tree/master/Data_Preprocess). After registration, the image dimension will be $182\times218\times182$ and the voxel dimension will be $1\times1\times1$ $mm^3$.



## Training
Run `python train.py` to train the network.

## Testing
Run `python test.py`.

## Contact
Email at: kushol@ualberta.ca

## Acknowledgement
This basic structure of the code relies on the project of [TransUNet](https://github.com/Beckschen/TransUNet)

[GFNet](https://github.com/raoyongming/GFNet)
