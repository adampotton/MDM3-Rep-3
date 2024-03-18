# Training log for feature extraction on the inception-V3 CNN architecture, pretrained on imagenet

The original network was trained on image net with 1000 categories and both a primary output layer and auxilary output layer, which is used during training. To attempt to extract features from the feature embeddings created by the model, these two final layers can be retrained so that there are 19 output categories, corresponding to different tree species in the training data.

To start we took a stratified sample of 10% of each species aerial image data from the dataset, including an even mix of tree stand ages. Because Inception-V3 is pretrained on RGB images, it only has 3 input channels, so I elected to remove the IR channel and just finetune with RGB training data.

## Training Run no. 1
-  **epochs**: 15
- **optimizer**: SGD with lr=0.001, momentum=0.9
- **batch_size**: 4
- **best val accuracy**: 0.383117

## Training Run no. 1
On this run I will increase the batch size and use an adam optimizer to try and improve the convergence. It looks like further epochs may lead to better performance. I should also look into the possibility of training an encoder to create a 3 channel feature embedding from 4 channel RGB-IR input data.