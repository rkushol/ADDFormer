# ADDFormer
Alzheimer's Disease Detection from structural MRI using Fusion Transformer.

The code will be released at the time of publication

## Abstract
Alzheimer's disease is the most prevalent neurodegenerative disorder characterized by degeneration of the brain. It is classified as a brain disease causing dementia that presents with memory loss and cognitive impairment. Experts primarily use brain imaging and other tests to rule out the disease. To automatically detect Alzheimer's patients from healthy controls, this study adopts the vision transformer architecture, which can effectively capture the global or long-range relationship of image features. To further enhance the network's performance, frequency and image domain features are fused together since MRI data is acquired in the frequency domain before being transformed to images. We train the model with selected coronal 2D slices to leverage the transfer learning property of pre-training the network using ImageNet. Finally, the majority voting of the coronal slices of an individual subject is used to generate the final classification score. Our proposed method has been evaluated on the publicly available benchmark dataset ADNI. The experimental results demonstrate the advantage of our proposed approach in terms of classification accuracy compared with that of the state-of-the-art methods.
