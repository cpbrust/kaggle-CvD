# kaggle-CvD
Cats vs Dogs Kaggle Competition

This code documents my foray into machine learning, as detailed on my blog (https://cpbrust.wordpress.com/). This readme summarizes the code and my contributions to it.

cnn.py:
This is code that I found by googling. It reads in the training set from the Kaggle CvD competition, pre-processes it, and trains a model, then saves that model.

cnnTest.py:
This loads the model we trained in cnn.py and evaluates it on the test set, storing the answers in a CSV. We'll pass this CSV into Mathematica next for some post-processing.

postprocessing.nb:
This Mathematica notebook handles the postprocessing of the output of cnnTest.py. It implements the regulation described in the associated blog post.

