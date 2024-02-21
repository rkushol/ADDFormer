# ADDFormer
Alzheimer's Disease Detection from structural MRI using Fusion Transformer.

The paper has been published in the Conference of ISBI 2022.
Link: (https://ieeexplore.ieee.org/iel7/9761376/9761399/09761421.pdf).

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


# SF2Former
Amyotrophic lateral sclerosis identification from multi-center MRI data using spatial and frequency fusion transformer

The paper has been published in the Journal of Computerized Medical Imaging and Graphics 2023.
Link: (https://www.sciencedirect.com/science/article/pii/S0895611123000976).

```
@article{kushol2023sf2former,
  title={SF2Former: Amyotrophic lateral sclerosis identification from multi-center MRI data using spatial and frequency fusion transformer},
  author={Kushol, Rafsanjany and Luk, Collin C and Dey, Avyarthana and Benatar, Michael and Briemberg, Hannah and Dionne, Annie and Dupr{\'e}, Nicolas and Frayne, Richard and Genge, Angela and Gibson, Summer and others},
  journal={Computerized medical imaging and graphics},
  volume={108},
  pages={102279},
  year={2023},
  publisher={Elsevier}
}
```

## Abstract
Amyotrophic Lateral Sclerosis (ALS) is a complex neurodegenerative disorder characterized by motor neuron degeneration. Significant research has begun to establish brain magnetic resonance imaging (MRI) as a potential biomarker to diagnose and monitor the state of the disease. Deep learning has emerged as a prominent class of machine learning algorithms in computer vision and has shown successful applications in various medical image analysis tasks. However, deep learning methods applied to neuroimaging have not achieved superior performance in classifying ALS patients from healthy controls due to insignificant structural changes correlated with pathological features. Thus, a critical challenge in deep models is to identify discriminative features from limited training data. To address this challenge, this study introduces a framework called SF2Former, which leverages the power of the vision transformer architecture to distinguish ALS subjects from the control group by exploiting the long-range relationships among image features. Additionally, spatial and frequency domain information is combined to enhance the network’s performance, as MRI scans are initially captured in the frequency domain and then converted to the spatial domain. The proposed framework is trained using a series of consecutive coronal slices and utilizes pre-trained weights from ImageNet through transfer learning. Finally, a majority voting scheme is employed on the coronal slices of each subject to generate the final classification decision. The proposed architecture is extensively evaluated with multi-modal neuroimaging data (i.e., T1-weighted, R2*, FLAIR) using two well-organized versions of the Canadian ALS Neuroimaging Consortium (CALSNIC) multi-center datasets. The experimental results demonstrate the superiority of the proposed strategy in terms of classification accuracy compared to several popular deep learning-based techniques.


## Requirements
PyTorch  
nibabel  
scipy  
scikit-image  


## Datasets
ADNI dataset can be downloaded from [ADNI](http://adni.loni.usc.edu/) (Alzheimer’s Disease Neuroimaging Initiative)


## Preprocessing
### Skull stripping using Freesurfer
Command ``recon-all -subjid subjid -i inputfile.nii -autorecon1``

Details can be found at https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all



## Training
Run `python train.py` to train the network.

## Testing
Run `python test.py`.

## Contact
Email at: kushol@ualberta.ca

## Acknowledgement
This basic structure of the code relies on the project of [TransUNet](https://github.com/Beckschen/TransUNet)

[GFNet](https://github.com/raoyongming/GFNet)
