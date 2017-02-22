# kaggle-CvD
Cats vs Dogs Kaggle Competition

This code documents my foray into machine learning, as detailed on my blog (https://cpbrust.wordpress.com/). This readme summarizes the code and my contributions to it.

cnn.py:
This is code that I found by googling. It reads in the training set from the Kaggle CvD competition, pre-processes it, and trains a model, then saves that model.

cnnTest.py:
This loads the model we trained in cnn.py and evaluates it on the test set, storing the answers in a CSV. We'll pass this CSV into Mathematica next for some post-processing.

postprocessing.nb:
This Mathematica notebook handles the postprocessing of the output of cnnTest.py. It implements the regulation described in the associated blog post.

cnn2.py:
The beginning of our experimentation. We tried increasing the input resolution to 96 x 96, increasing the number of filters, and changing the learning algorithm to try to combat overfitting.

cnn3.py:
We tried using ELUs instead of ReLUs, and explicit L2 regularization, to no avail.

cnn4.py:
We revert to our original architecture, but make the network deeper, with more hidden layers.

cnn5.py:
We modify v4 further, reincluding L2 regularization, adding more filters in the last two convolutional layers, and trying out a sigmoid activation function in the final layer.

cnn6.py:
We return to v4 again, add another hidden layer, and increase the size of the filters used in convolution from 3 to 5 pixels.

cnn7.py:
We design the deepest architecture that can be managed while simultaneously allowing for features at all and not exhausting our RAM. This amounts to 10 convolutional layers. This is probably as good as we're gonna get out of this desktop.
