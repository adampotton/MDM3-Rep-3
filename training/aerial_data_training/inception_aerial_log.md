# Training log for feature extraction on the inception-V3 CNN architecture, pretrained on imagenet

The original network was trained on image net with 1000 categories and both a primary output layer and auxilary output layer, which is used during training. To attempt to extract features from the feature embeddings created by the model, these two final layers can be retrained so that there are 19 output categories, corresponding to different tree species in the training data.

To start we took a stratified sample of 10% of each species aerial image data from the dataset, including an even mix of tree stand ages. Because Inception-V3 is pretrained on RGB images, it only has 3 input channels, so I elected to remove the IR channel and just finetune with RGB training data.

## Training Run no. 1
-  **epochs**: 15
- **optimizer**: SGD with lr=0.001, momentum=0.9
- **batch_size**: 4
- **best val accuracy**: 0.383117
**Number 1.5**
- **batch_size**: 32
- **best val accuracy**: 0.45
- **test accuracy**: 0.42

## Training Run no. 2
For this run I used a larger batch size and added a 1x1x4 convolutional layer to the start of the network to reduce the 4 channel input images to 3 channels to be used as input to the inception-v3 model
-  **epochs**: 30
- **optimizer**: SGD with lr=0.001, momentum=0.9
- **batch_size**: 32
- **best val accuracy**: 0.463203
- **best train accuracy**: 0.4585
- **best val loss**: 1.8348
- **best train loss**: 2.3460


## TODO: 
- tune SGD hyperparameters
- initialize final layer probabilities in line with the training data percentages
- Determine if the current p=0.5 dropout is optimal

On this run I will increase the batch size and tune the SGD hyperparameters (SGD is almost always better for ConvNets) to try and improve the convergence. It looks like further epochs may lead to better performance. I should also look into the possibility of training an encoder to create a 3 channel feature embedding from 4 channel RGB-IR input data. Additionally I should try to initialize the final layer probabilities in line with the training data percentages.

Potentially investigate the use of dropout, although it doesnt play nice with normalization and I am not certain if it is appropriate in tandem with the inception-V3 architecture.




# Notes on Inception V3 architecture
- Before the final linear layer, dropout is implemented with p=0.5 (quite aggressive). Look into this.

# Additional Notes:
- Setting the random seed can be quite important and can lead to a large amount of variation in the training performance. (Sobol Sequences?)