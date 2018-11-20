# trainEagerMnist

Example of training mnist in tensorflow eager mode.

I modified the code from [mnist_eager](https://github.com/tensorflow/models/blob/master/official/mnist/mnist_eager.py) 
and [07_convolutional_neural_networks_for_emotion_recognition](https://github.com/madalinabuzau/tensorflow-eager-tutorials/blob/master/07_convolutional_neural_networks_for_emotion_recognition.ipynb).

When eagerFlag is 0, do not use eager mode, and do not add tfe.enable_eager_execution(), otherwise the code can not run. 

When eagerFlag is 1 or 2, we use different ways of eager mode.