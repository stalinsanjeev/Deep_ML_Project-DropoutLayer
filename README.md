# Deep_ML_Project-
Code for Project on Dropout layers for 576 course

The effects of dropout layers were tested in this project with several Datasets. Expriments were done to see if adding an dropout layer would prevent overfitting. Image and text was used for classification and Regeneration.

## AutoEncoders 

  In Autoencoders.ipynb experimentations were done to reconstruct the images using autoencoders with and without dropouts. Autoencoders work with introducing noise to the given dataset and then they try to recreate the dataset using the images which are filled with noise [5]. Several variations of a dense autoencoder model are constructed. Each model is a Sequential model consisting of Dense layers and Dropout layers. The architecture varies the number of neurons and dropout rates in different configurations. Example architecture includes layers like Dense(128, activation='relu') and Dropout(dropout_rate). Models are compiled with the 'adam' optimizer and 'binary_crossentropyloss function.

  The architecture ends with a dense layer which has a sigmoid activation function to reconstruct the input images. Sequential model was used with three dense layers and three dropout layers.Since MNIST dataset is used for autoencoders the dropout rates were tested from 0,0.01,0.04,0.05,0.1,0.2,0.30.4,0.5 and 0.75

## MNIST 

  The MNIST_0.2 and MNIST_0.5 noteboonks consist of sequential model with 3 dense layers and 2 dropout layers . A Flatten
layer is used at first to convert the input into a one dimensional array. Then a fully connected dense layers is introduced with 128 neurons (or units) using the ReLU (Rectified Linear Unit) activation function. After which a dropout layer is introduced with variable dropouts rates changing through the experiment followed by another dense layer and dropout layer. Finally a Dense layer with 10 neurons using softmax activation function is used for 10 class classification.
Dropouts rates with 0.2 and 0.5 were tested for the model. Models without dropout rate were tested to access the performance. Epochs of 5,10,15,20 and 50 were used to see the effects of overfitting.

## CIFAR 

  In CIFAR-CNN_4.pynb notebook a CNN was used with convolutional and dense layers consisting of 3 Conv2D layers, 2 MaxPooling2D layers, 3 Dropout layers, 1 Flatten layer, and 2 Dense layers. Dropout rates of 0,0.5 and 0.2 with epochs of 10,15,20 were used for this dataset

## RCV 

  In RCV_1.pynb and RCV_2.pynb notebooks The Reuters Corpus Volume 1
dataset was used to check the effect of
dropouts on text. A simple model of 4
layers consisting LSTM were used.
First an embedding layer designed to
convert input sequences of integers
(representing words) into dense vectors
of fixed size (64 in this case ) was
introduced followed by LSTM layer of
100 units capable of learning long term
dependencies were used. The
return_sequences=True parameter was used which means that this layer will return the full sequence of outputs for each input sequence, which is necessary for stacking another LSTM layer on top of it [6]. Another layer of LSTM without return_sequences=True parameter was used which will will only return the output of the last time step Followed by a dense layer of 46 Neurons was using the softmax activation function was used.

## SVHN

  In Google_svhn.pynb notebook is a simple sequential model with three dense layers and two dropouts layers having a dropout rate of 0.2 between them. This was used to see the effect of dropouts on sreet view images of house numbers collected from google database.
