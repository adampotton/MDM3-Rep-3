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

## Training Run no. 3
For this run I used a larger training and val set with images filtered so that they contain >95% one species. This improved training accuracy significantly. Additionally I decided to use the default imagenet normalisation values for the RGB channels and calculate new values for the IR channel (this was also the easiest solution due to memory limitations).
- **epochs**: 100
- **optimizer**: AdamW with lr=0.01,betas=[0.8,0.99],w_decay=0.001
- **batch_size**: 32
- **best val accuracy**: 0.614642
- **best train accuracy**: 0.4752
- **best val loss**: 2.103
Training was not very stable using these hyperparameters so I decided to run another Random Search with more hyperparameters and on the new data.


## Random Search
- Looks like there is not a direct correspondence between the loss function and the validation accuracy, perhaps validating that a probabilistic approach is preferable
- some training runs appear to converge to a minimum in less than 5 epochs and then fail to converge further
- There was some variation accross 20 samples but I think there is an underlying issue with the data/ loss function

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

**Inception_v3 optimizer (trained on image net)**:
- SGD
- batch size = 32
- Initial training using Momentum
- best results using RMSProp with momentum decay=0.9 and eps=1
- learning rate of 0.045
- decay every two epochs using an exponential rate of 0.94
- gradient clipping with threshold 2 was also used to stabilize training.

This paper was published before adam's rise to popularity so it seems reasonable to use Adam as the optimizer

**AdamW Hyperparameters**:
- beta1: exponential decay rate for the first moment. First moment is an exponential moving average of the gradients (commonly beta1 = 0.9)
- beta2: exponential decay rate for the second moment. The second moment is an exponential moving average of the squared gradients and gives an estimate of the variance in the local gradients. (commonly beta2 = 0.999)
- Typically the default values of betas work well but are sometimes adjusted.
- learning rate: learning rate is adaptive for adam but a typical value for the base learning rate is 0.001. Adam is very sensitive to learning rate. 
- A typical value for the weight decay coefficient is 0.01

**Selected Hyperparameter ranges For Random Search**
Learning Rate (α):
[0.0001,0.1]
Since the dataset is relatively small using larger learning rates is likely to introduce instabilities 

β1 (Exponential Decay for First Moment Estimation):
Common range: [0.8, 0.999].

β2 (Exponential Decay for Second Moment Estimation):
Common range: [0.9, 0.999].

Weight Decay:
[0.0001,0.1]
penalizes large weights

**Random Search Procedure**
Infinite loop which trains the model and reports validation loss in the following manner for each loop iteration
1. Randomly generated a value for each of the hyperparameters (lr,b1,b2,lambda)
2. Initialize an adamW optimizer with the generated hyperparameters
3. train the model using the optimizer
4. Once training appears to reach a minimum terminate and return model performance statistics
5. write statistics, number of epochs and associated hyperparameters to a new line in a csv or text file 
6. repeat

We can implement early stopping during the training of each model to ensure that 1. the models are not overfit and 2. the training does not take an excessively long time