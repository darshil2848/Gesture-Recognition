## Import all modules
import numpy as np
import os
import cv2
from cv2 import imread, resize
import matplotlib.pyplot as plt
import random as rn
from keras import backend as K
import tensorflow as tf
import datetime
import os
from math import ceil

from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from keras.models import Sequential, Model
from keras.layers import Dense, GRU, Flatten, TimeDistributed, Flatten, BatchNormalization, Activation, Dropout
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers


"""
- We are going to try different approaches to tackle this problem. 
- All the approaches will require the data to be in a certain way. Therefore, we've opted to create an abstract base class, Generator.
- Any generator we write will inherit from this class and they will implement the mandatory virtual functions for them to work.

Will be writing the function of each method before the definition of the abstract class itself. 
If there are any new methods needed for a child class, some comments will be present before its definition. 
"""


class Generator(keras.utils.Sequence):
    def __init__(self):
        pass

    """ Main logic to get one batch of data is implemented in this method.
     @returns: (batch_data, batch_labels). 
     The shape of batch_data depends on the type of generator getting used. 
     Regardless, the first dimension will be batch_size.
     batch_labels.shape = (batch_size, numClasses)
    """
    def getBatchData(self, batch, currBatchSize):
        pass

    # Since all variables are public, this is not really needed. But keeping it anyway.
    # @return: Number of batches to run: depends on total number of videos/batch size
    def getNumBatches(self):
        pass

    # Since we are inheriting from keras.utils.Sequence, we need to implement __getitem__ & __len__
    # model.fit will call __getitem__ to get one batch of data. In our implementation, this acts like a wrapper around
    # getBatchData above
    def __getitem__(self, batchIdx):
        pass

    # Returns number of batches.
    def __len__(self):
        pass


"""
Approach 1
- So we use a pretrained CNN model like inception-v3 to extract features & then feed them to RNN. 
- This is the generator for this purpose: CNNRNNGenerator
"""


class CNNRNNGenerator(Generator):
    def __init__(self,
                 frameIdxList,
                 width=224,
                 height=224,
                 source_path=r"D:\DDownloads\UpGrad\NeuralNetwork\CaseStudy\Project_data\train",
                 batch_size=30,
                 dataCSV=r"D:\DDownloads\UpGrad\NeuralNetwork\CaseStudy\Project_data\train.csv",
                 numClasses=5,
                 numFeatures=2048,  # as per inception-v3
                 ):
        super().__init__()
        # Shuffle the data and store in a list
        self.frameIdxList = frameIdxList
        self.numFramesInVideo = len(frameIdxList)
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.source_path = source_path
        self.dataCSV = dataCSV
        self.numClasses = numClasses
        self.numFeatures = numFeatures

        self.data_doc = np.random.permutation(open(self.dataCSV).readlines())

        # Get vector list
        self.vectorList = np.random.permutation(self.data_doc)
        self.numVideos = len(self.vectorList)
        self.remBatchSize = self.numVideos % self.batch_size
        self.numBatches = ceil(self.numVideos / self.batch_size)
        self.currBatchIdx = 0
        self.numChannels = 3
        self.featureExtractor = None

    # Build the feature extractor(inception-v3 in our case
    def __build_feature_extractor(self, preTrainedModel="inceptionv3"):
        featureExtractor = None
        if preTrainedModel == "inceptionv3":
            # Disable output layer, so when we do model.predict, it only returns the feature map and the class
            # probabilities. I think imagenet has 1000 classes, we don't need their probabilities.
            featureExtractor = keras.applications.InceptionV3(
                weights="imagenet",
                include_top=False,
                pooling="avg",
                input_shape=(self.width, self.height, self.numChannels))
            # Get preprocess_input from inception_v3 module.
            # This is important because it is responsible for normalizing and converting the image to the format with
            # which inception_v3 was trained on.
            preprocess_input = keras.applications.inception_v3.preprocess_input
            inputs = keras.Input((self.width, self.height, self.numChannels))
            preprocessed = preprocess_input(inputs)
            outputs = featureExtractor(preprocessed)
            return keras.Model(inputs, outputs, name="feature_extractor")
        # Try other feature extractors, if needed
        elif preTrainedModel == "vgg16":
            pass
        else:
            pass

    def getNumBatches(self):
        return self.numBatches

    # Since we are using inception-v3 to extract features, we'll initialize it with this method
    def initializeFeatureExtractor(self, preTrainedModel="inceptionv3"):
        self.featureExtractor = self.__build_feature_extractor(preTrainedModel)

    def getBatchData(self, batch, currBatchSize):
        # The input to RNN will be the output of CNN(inceptionv3)
        # The output of inceptionv3 will be 2048. Since we have 30 frames(numFramesInVideo), batch_data: 30, 2048.
        # Finally, every batch will have this input.
        batch_data = np.zeros((self.batch_size, self.numFramesInVideo, self.numFeatures))
        batch_labels = np.zeros((self.batch_size, self.numFramesInVideo, self.numClasses))
        for batchIdx in range(currBatchSize):
            # Get the vector name (or directory name)
            unprocessedCurrVectorString = self.vectorList[batchIdx + (batch * self.batch_size)].strip()
            vectorName = unprocessedCurrVectorString.split(";")[0]
            # Store vector path
            vectorDir = self.source_path + "\\" + vectorName

            # List all the frames within the directory(vectorName). # Read each image one by one
            allFrames = os.listdir(vectorDir)

            # videoData has all the images imread
            videoData = np.zeros((self.numFramesInVideo, self.width, self.height, self.numChannels))

            # Its corresponding label
            videoLabel = np.zeros((self.numFramesInVideo, self.numClasses))

            # Current video's frames loop
            for frameIdx in self.frameIdxList:
                # Get path of current frame using frameIdx
                currentFrame = vectorDir + "\\" + allFrames[frameIdx]
                # Read image
                frame = cv2.imread(currentFrame).astype(np.float32)
                # Resize the image to what the feature extractor wants
                resizedCurrentFrame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
                # Store frame in videoData tensor.
                videoData[frameIdx,] = resizedCurrentFrame
                # Set the corresponding class' index to 1: One hot encoding
                videoLabel[frameIdx, int(unprocessedCurrVectorString.split(";")[2])] = 1

            batch_data[batchIdx,] = self.featureExtractor.predict(videoData)
            batch_labels[batchIdx,] = videoLabel
        return batch_data, batch_labels

        # Return number of batches in dataset. Needed as we are inheriting from keras.utils.Sequence

    def __len__(self):
        return self.numBatches

    # Get one batch of data. Needed as we are inheriting from keras.utils.Sequence

    # Iterate over batches. next(generator) will not be called during model.fit. __getitem__ gets called
    def __getitem__(self, batchIdx):
        # Adjusting batch size as per batchIdx. Remember the last batch may not have enough data to be equal to
        # numVideos/batch_size, it will be equal to numVideos % batch_size.
        currBatchSize = self.batch_size if batchIdx < (self.numVideos // self.batch_size) else self.remBatchSize
        batch_data, batch_labels = self.getBatchData(batchIdx, currBatchSize)
        return batch_data, batch_labels

    """ Keeping this for debugging purposes. Not really needed for the functioning of our code.
        We used it to test generator with next(generator).
    """
    def __next__(self):
        batch_data = None
        batch_labels = None
        print(self.currBatchIdx)
        if self.currBatchIdx < self.__len__():
            batch_data, batch_labels = self.__getitem__(self.currBatchIdx)
            self.currBatchIdx += 1
        else:
            self.currBatchIdx = 0
            batch_data, batch_labels = self.__getitem__(self.currBatchIdx)
        return batch_data, batch_labels


"""
Approach 2
- Here we create a custom Conv3D network.
- The input to this network will be the images itself.
Generator:Conv3DGenerator
"""


class Conv3DGenerator(Generator):
    def __init__(self,
                 frameIdxList,
                 width=224,
                 height=224,
                 source_path=r"D:\DDownloads\UpGrad\NeuralNetwork\CaseStudy\Project_data\train",
                 batch_size=30,
                 dataCSV=r"D:\DDownloads\UpGrad\NeuralNetwork\CaseStudy\Project_data\train.csv",
                 numClasses=5):
        super().__init__()
        # Shuffle the data and store in a list
        self.frameIdxList = frameIdxList
        self.numFramesInVideo = len(frameIdxList)
        self.batch_size = batch_size
        self.width = width
        self.height = height

        self.source_path = source_path
        self.dataCSV = dataCSV
        self.numClasses = numClasses
        self.data_doc = np.random.permutation(open(self.dataCSV).readlines())

        # Get vector list
        self.vectorList = np.random.permutation(self.data_doc)
        self.numVideos = len(self.vectorList)
        self.remBatchSize = self.numVideos % self.batch_size
        self.numBatches = ceil(self.numVideos / self.batch_size)
        self.currBatchIdx = 0
        self.numChannels = 3

    def getNumBatches(self):
        return self.numBatches

    def __len__(self):
        return self.numBatches

    def getBatchData(self, batch, currBatchSize):
        # The input to Conv3d will just be the frames of each video
        batch_data = np.zeros((self.batch_size, self.numFramesInVideo, self.width, self.height, self.numChannels))
        batch_labels = np.zeros((self.batch_size, self.numClasses))
        for batchIdx in range(currBatchSize):
            # Get the vector name (or directory name)
            unprocessedCurrVectorString = self.vectorList[batchIdx + (batch * self.batch_size)].strip()
            vectorName = unprocessedCurrVectorString.split(";")[0]
            # Store vector path
            vectorDir = self.source_path + "\\" + vectorName

            # List all the frames within the directory(vectorName). # Read each image one by one
            allFrames = os.listdir(vectorDir)

            # videoData has all the images imread
            videoData = np.zeros((self.numFramesInVideo, self.width, self.height, self.numChannels))

            # Its corresponding label
            videoLabel = np.zeros((self.numFramesInVideo, self.numClasses))

            # Current video's frames loop
            for frameIdx in self.frameIdxList:
                # Get path of current frame using frameIdx
                currentFrame = vectorDir + "\\" + allFrames[frameIdx]
                # Read image
                frame = cv2.imread(currentFrame).astype(np.float32)

                # Resize the image to (224, 224, 3).
                resizedCurrentFrame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

                # Normalize
                resizedCurrentFrame = resizedCurrentFrame / 255.0
                # Store frame in videoData tensor.
                videoData[frameIdx,] = resizedCurrentFrame

            batch_data[batchIdx,] = videoData
            # Set the corresponding class' index to 1: One hot encoding
            batch_labels[batchIdx, int(unprocessedCurrVectorString.split(";")[2])] = 1
        return batch_data, batch_labels

    # Iterate over batches. next(generator) will not be called during model.fit. __getitem__ gets called
    def __getitem__(self, batchIdx):
        # Adjusting batch size as per batchIdx. Remember the last batch may not have enough data to be equal to
        # numVideos/batch_size, it will be equal to numVideos % batch_size.
        currBatchSize = self.batch_size if batchIdx < (self.numVideos // self.batch_size) else self.remBatchSize
        batch_data, batch_labels = self.getBatchData(batchIdx, currBatchSize)
        return batch_data, batch_labels

    """ Keeping this for debugging purposes. Not really needed for the functioning of our code.
            We used it to test generator with next(generator).
        """
    def __next__(self):
        batch_data = None
        batch_labels = None
        print(self.currBatchIdx)
        if self.currBatchIdx < self.__len__():
            batch_data, batch_labels = self.__getitem__(self.currBatchIdx)
            self.currBatchIdx += 1
        else:
            self.currBatchIdx = 0
            batch_data, batch_labels = self.__getitem__(self.currBatchIdx)
        return batch_data, batch_labels


"""
Approach 2
- Here we create a custom Conv3D network like the Conv3DGenerator, only difference being the input is not the images, rather 
  we obtain a dense optical flow output of the images and feed that to a Conv3D network to predict the classes
Generator:OpticalFlowGenerator
"""

# The constructor differs slighly from the generators' constructor:
# We can use an existing vector list(which has already been shuffled). In such scenarios, we provide two more arguments:
# useVectorList(a boolean) and the actual vectorList to be used.


class OpticalFlowGenerator(Generator):
    def __init__(self,
                 frameIdxList,
                 width=224,
                 height=224,
                 source_path=r"D:\DDownloads\UpGrad\NeuralNetwork\CaseStudy\Project_data\train",
                 batch_size=30,
                 dataCSV=r"D:\DDownloads\UpGrad\NeuralNetwork\CaseStudy\Project_data\train.csv",
                 numClasses=5,
                 useVectorList=False,
                 vectorList=None):
        super().__init__()
        # Shuffle the data and store in a list
        self.frameIdxList = frameIdxList
        self.numFramesInVideo = len(frameIdxList)
        self.batch_size = batch_size
        self.width = width
        self.height = height

        self.source_path = source_path
        self.dataCSV = dataCSV
        self.numClasses = numClasses
        self.data_doc = np.random.permutation(open(self.dataCSV).readlines())

        # Get vector list
        if useVectorList:
            self.vectorList = vectorList
        else:
            self.vectorList = np.random.permutation(self.data_doc)
        self.numVideos = len(self.vectorList)
        self.remBatchSize = self.numVideos % self.batch_size
        self.numBatches = ceil(self.numVideos / self.batch_size)
        self.currBatchIdx = 0
        self.numChannels = 3

    def getNumBatches(self):
        return self.numBatches

    def __len__(self):
        return self.numBatches

    """
    - Take current and next frame's absolute paths as input arguments.
    - imread the frames
    - resize to desire size
    - convert to gray(since OF only needs luma component)
    - Get dense OF output using Gunnar Farneback's method
    - Return OF output
    """
    def getOFOutput(self, currFramePath, nextFramePath):
        # Read images
        currFrame = cv2.imread(currFramePath)
        nextFrame = cv2.imread(nextFramePath)

        # Resize to an agreed upon size(Inception wants 224, 224)
        resizedCurrentFrame = cv2.resize(currFrame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        resizedNextFrame = cv2.resize(nextFrame, (self.width, self.height), interpolation=cv2.INTER_AREA)

        # Get the gray versions
        currFrameGray = cv2.cvtColor(resizedCurrentFrame, cv2.COLOR_BGR2GRAY)
        nextFrameGray = cv2.cvtColor(resizedNextFrame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev=currFrameGray, next=nextFrameGray, flow=None, pyr_scale=0.5,
                                            levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        return flow

    def getBatchData(self, batch, currBatchSize):
        # The input to Conv3d will be the flow output of each frame
        # The last dim is 2 because flow is a vector of dx/dt, dy/dt
        batch_data = np.zeros((self.batch_size, self.numFramesInVideo - 1, self.width, self.height, 2))
        batch_labels = np.zeros((self.batch_size, self.numClasses))
        for batchIdx in range(currBatchSize):
            # Get the vector name (or directory name)
            unprocessedCurrVectorString = self.vectorList[batchIdx + (batch * self.batch_size)].strip()
            vectorName = unprocessedCurrVectorString.split(";")[0]
            # Store vector path
            vectorDir = self.source_path + "\\" + vectorName

            # List all the frames within the directory(vectorName). # Read each image one by one
            allFrames = os.listdir(vectorDir)

            # videoData has all the images' flow outputs
            videoData = np.zeros((self.numFramesInVideo - 1, self.width, self.height,
                                  2))  # The last dim is 2 because flow is a vector of dx/dt, dy/dt

            # Current video's frames loop
            frameIdx = 0
            while frameIdx < (
                    self.numFramesInVideo - 1):  # Flow will have one less frame since it needs (prev, next) pairs
                # Get path of current frame using frameIdx
                currentFrame = vectorDir + "\\" + allFrames[frameIdx]

                # Get path of next frame using frameIdx
                nextFramePath = vectorDir + "\\" + allFrames[frameIdx + 1]

                flow = self.getOFOutput(currFramePath=currentFrame, nextFramePath=nextFramePath)
                videoData[frameIdx,] = flow
                frameIdx += 1
            batch_data[batchIdx,] = videoData
            # Set the corresponding class' index to 1: One hot encoding
            batch_labels[batchIdx, int(unprocessedCurrVectorString.split(";")[2])] = 1
        return batch_data, batch_labels

    # Iterate over batches. next(generator) will not be called during model.fit. __getitem__ gets called
    def __getitem__(self, batchIdx):
        # Adjusting batch size as per batchIdx. Remember the last batch may not have enough data to be equal to
        # numVideos/batch_size, it will be equal to numVideos % batch_size.
        currBatchSize = self.batch_size if batchIdx < (self.numVideos // self.batch_size) else self.remBatchSize
        batch_data, batch_labels = self.getBatchData(batchIdx, currBatchSize)
        return batch_data, batch_labels

    # Keeping this for debugging purposes. Not really needed, was used to test: next(generator)
    def __next__(self):
        batch_data = None
        batch_labels = None
        print(self.currBatchIdx)
        if self.currBatchIdx < self.__len__():
            batch_data, batch_labels = self.__getitem__(self.currBatchIdx)
            self.currBatchIdx += 1
        else:
            self.currBatchIdx = 0
            batch_data, batch_labels = self.__getitem__(self.currBatchIdx)
        return batch_data, batch_labels


## Model utilities
## -- rnn -- ##
def get_rnn_model(numFrames=30, numFeatures=2048):
    frame_features_input = keras.Input(shape=(numFrames, numFeatures))
    x = GRU(16, return_sequences=True)(frame_features_input)
    # x = GRU(8, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)  # Adding one dense layer
    dense = Dense(5, activation="softmax")
    outputs = TimeDistributed(dense)(x)
    rnn_model = keras.Model(inputs=frame_features_input, outputs=outputs)
    rnn_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return rnn_model


def create_model_dir():
    curr_dt_time = datetime.datetime.now()
    model_name = r"rcnn\rcnn_init" + '_' + str(curr_dt_time).replace(' ', '').replace(':', '_') + '\\'
    currDir = os.getcwd()
    if not os.path.exists(currDir + "\\" + "rcnn"):
        os.mkdir(currDir + "\\" + "rcnn")

    modelPath = currDir + "\\" + model_name
    if not os.path.exists(modelPath):
        os.mkdir(modelPath)

    filepath = modelPath + 'rcnn-{epoch:05d}-{loss:.5f}-{accuracy:.5f}-{val_loss:.5f}-{val_accuracy:.5f}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, save_weights_only=False,
                                 mode='auto', period=1)
    LR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2,
                           min_lr=1e-4)  # write the REducelronplateau code here
    return [checkpoint, LR]


## -- rnn  end-- ##

## -- Conv3d -- ##
def get_conv3d_model(numFeaturesInFirstLayer, numFrames, input_shape, output_activation="softmax",
                     numClasses=5, numNeuronsInDenseLayer=256):
    # Model
    model = models.Sequential()

    # Convolution layer with `numFeaturesInFirstLayer` features, 3x3 filter and a relu activation with 2x2 pooling
    model.add(layers.Conv3D(numFeaturesInFirstLayer, (3, 3, 3), padding='same', activation='relu',
                            input_shape=(numFrames, input_shape[0], input_shape[1], input_shape[2])))
    model.add(layers.MaxPooling3D())

    # Convolution layer with `numFeaturesInFirstLayer * 2` features, 3x3 filter and relu activation with 2x2 pooling
    model.add(layers.Conv3D((numFeaturesInFirstLayer * 2), (3, 3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling3D())

    # Convolution layer with `numFeaturesInFirstLayer * 4` features, 3x3 filter and relu activation with 2x2 pooling
    model.add(layers.Conv3D((numFeaturesInFirstLayer * 4), (3, 3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling3D())
    # model.add(Dropout(0.25))

    # Convolution layer with `numFeaturesInFirstLayer * 8` features, 3x3 filter and relu activation with 2x2 pooling
    model.add(layers.Conv3D((numFeaturesInFirstLayer * 8), (3, 3, 3), padding='same', activation='relu',
                            kernel_regularizer=regularizers.l2(l=0.01)))
    model.add(layers.MaxPooling3D())

    model.add(layers.Flatten())
    model.add(layers.Dense(numNeuronsInDenseLayer, activation='relu'))
    model.add(Dropout(0.4))
    model.add(layers.Dense(numClasses, activation=output_activation))

    optimiser = "adam"
    model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    print(model.summary())
    return model


def get_conv3d_callbacks():
    curr_dt_time = datetime.datetime.now()
    # for OF:
    model_name = r"conv3d_of\conv3d_init" + '_' + str(curr_dt_time).replace(' ', '').replace(':', '_') + '\\'
    # For conv3d
    # model_name = r"conv3d_of\conv3d_init" + '_' + str(curr_dt_time).replace(' ', '').replace(':', '_') + '\\'
    # For cnnrnn
    # model_name = r"cnnrnn\conv3d_init" + '_' + str(curr_dt_time).replace(' ', '').replace(':', '_') + '\\'
    currDir = os.getcwd()
    if not os.path.exists(currDir + "\\" + "conv3d_of"):
        os.mkdir(currDir + "\\" + "conv3d_of")

    modelPath = currDir + "\\" + model_name
    if not os.path.exists(modelPath):
        os.mkdir(modelPath)

    filepath = modelPath + 'conv3d-{epoch:05d}-{loss:.5f}-{categorical_accuracy:.5f}-{val_loss:.5f}-{val_categorical_accuracy:.5f}.h5'
    # val_loss
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, save_weights_only=False,
                                 mode='auto', period=1)
    LR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2,
                           min_lr=1e-4)  # write the Reducelronplateau code here
    return [checkpoint, LR]
    # return [LR]


## -- Conv3d done -- ##
def initializeSeeds():
    np.random.seed(30)
    rn.seed(30)
    tf.random.set_seed(30)


def driver():
    initializeSeeds()

    # Some constants
    frameIdxList = list(range(0, 30))
    width = 224
    height = 224
    numChannels = 3
    nClasses = 5
    batch_size = 10
    num_epochs = 10

    val_path = r"D:\DDownloads\UpGrad\NeuralNetwork\CaseStudy\Project_data\val"
    val_csv = r"D:\DDownloads\UpGrad\NeuralNetwork\CaseStudy\Project_data\val.csv"

    ## Type of model: Modify this.
    # modelType = "conv3d"
    # modelType = "OFGen"
    modelType = "OF+Conv3d"
    numFeaturesInFirstLayer = 16
    numNeuronsInDenseLayer = 128

    # Switch case based on the type of the approach you want to run.
    if modelType == "cnn-rnn":
        ## ------------------------ CNN-RNN ------------------------------ ##
        # Generators
        train_gen = CNNRNNGenerator(frameIdxList=frameIdxList, batch_size=batch_size)
        train_gen.initializeFeatureExtractor()

        val_gen = CNNRNNGenerator(frameIdxList=frameIdxList, source_path=val_path, batch_size=batch_size,
                                  dataCSV=val_csv)
        val_gen.initializeFeatureExtractor()

        # Model
        model = get_rnn_model()
        print(model.summary())
        callbacks_list = create_model_dir()

        """
        loaded_model = models.load_model(
            r"D:\PyCharm\Projects\GestureRecognition\rcnn\rcnn_init_2022-12-3110_32_14.457700\rcnn-00009-0.59062-0.71716"
            r"-0.74358-0.68133.h5")
        loaded_model.fit(train_gen,
                         steps_per_epoch=train_gen.getNumBatches(),
                         epochs=num_epochs,
                         verbose=1,
                         validation_data=val_gen,
                         validation_steps=val_gen.getNumBatches(),
                         workers=6,
                         use_multiprocessing=False,
                         initial_epoch=0,
                         callbacks=callbacks_list)
        """
        # Training model
        model.fit(train_gen,
                  steps_per_epoch=train_gen.getNumBatches(),
                  epochs=num_epochs,
                  verbose=1,
                  validation_data=val_gen,
                  validation_steps=val_gen.getNumBatches(),
                  workers=6,
                  use_multiprocessing=False,
                  initial_epoch=0,
                  callbacks=callbacks_list)
    elif modelType == "conv3d":
        ## ------------------------ Conv3D ----------------------------- ##
        convTrainGen = Conv3DGenerator(frameIdxList=frameIdxList, batch_size=batch_size)
        convValGen = Conv3DGenerator(frameIdxList=frameIdxList, source_path=val_path, batch_size=batch_size,
                                     dataCSV=val_csv)

        conv3dModel = get_conv3d_model(numFeaturesInFirstLayer=numFeaturesInFirstLayer, numFrames=len(frameIdxList),
                                       numNeuronsInDenseLayer=numNeuronsInDenseLayer)

        callbacks_list = get_conv3d_callbacks()

        testTrainedModel = True
        if testTrainedModel:
            loaded_model = models.load_model(
                r"D:\PyCharm\Projects\GestureRecognition\conv3d\kernelRegularizer_conv3d_init_2023-01-0101_23_20.243618\conv3d-00009-0.21692-0.92836-0.54015-0.88000.h5")
            loaded_model.fit(convTrainGen,
                             steps_per_epoch=convTrainGen.getNumBatches(),
                             epochs=num_epochs,
                             verbose=1,
                             validation_data=convValGen,
                             validation_steps=convValGen.getNumBatches(),
                             workers=4,
                             class_weight=None,
                             initial_epoch=0,
                             callbacks=callbacks_list)
        else:
            history = conv3dModel.fit(convTrainGen,
                                      steps_per_epoch=convTrainGen.getNumBatches(),
                                      epochs=num_epochs,
                                      verbose=1,
                                      validation_data=convValGen,
                                      validation_steps=convValGen.getNumBatches(),
                                      workers=4,
                                      class_weight=None,
                                      initial_epoch=0,
                                      callbacks=callbacks_list)
    elif modelType == "OFGen":
        trainOFGen = OpticalFlowGenerator(frameIdxList=frameIdxList, batch_size=batch_size)

        valOFGen = OpticalFlowGenerator(frameIdxList=frameIdxList, source_path=val_path, batch_size=batch_size,
                                        dataCSV=val_csv)

        input_shape = (width, height, 2)  # Based on flow output
        conv3dModel = get_conv3d_model(numFeaturesInFirstLayer=numFeaturesInFirstLayer,
                                       numFrames=(len(frameIdxList) - 1),
                                       input_shape=input_shape, numNeuronsInDenseLayer=numNeuronsInDenseLayer)
        callbacks_list = get_conv3d_callbacks()
        testTrainedModel = True
        if testTrainedModel:
            loaded_model = models.load_model(
                r"D:\PyCharm\Projects\GestureRecognition\conv3d_of\conv3d_init_2023-01-0213_04_57.758259\conv3d-00005-0.12532-0.96866-0.27041-0.93000.h5")

            history = loaded_model.fit(trainOFGen,
                                       steps_per_epoch=trainOFGen.getNumBatches(),
                                       epochs=num_epochs,
                                       verbose=1,
                                       validation_data=valOFGen,
                                       validation_steps=valOFGen.getNumBatches(),
                                       workers=4,
                                       class_weight=None,
                                       initial_epoch=0,
                                       callbacks=callbacks_list)

        else:
            history = conv3dModel.fit(trainOFGen,
                                      steps_per_epoch=trainOFGen.getNumBatches(),
                                      epochs=num_epochs,
                                      verbose=1,
                                      validation_data=valOFGen,
                                      validation_steps=valOFGen.getNumBatches(),
                                      workers=4,
                                      class_weight=None,
                                      initial_epoch=0,
                                      callbacks=callbacks_list)
    elif modelType == "OF+Conv3d":
        """
        - Based on the paper: https://arxiv.org/abs/1705.07750, we take two models: Conv3d and OF->Conv3d and cache their predictions
        - Then we average them to obtain a new result. 
        - The added complexity gives a 1% boost in accuracy. 
        - With our implementation, may not be entirely worth it. But with some tweaks and changes, it can probably yield better results
        """
        # Get val generator for Conv3d
        convValGen = Conv3DGenerator(frameIdxList=frameIdxList, source_path=val_path, batch_size=batch_size,
                                     dataCSV=val_csv)
        # Get val generator for OF-Conv3d. Use the same data_doc as above
        valOFGen = OpticalFlowGenerator(frameIdxList=frameIdxList, source_path=val_path, batch_size=batch_size,
                                        dataCSV=val_csv, useVectorList=True, vectorList=convValGen.vectorList)

        ofModel = models.load_model(
            r"D:\PyCharm\Projects\GestureRecognition\conv3d_of\conv3d_init_2023-01-0213_04_57.758259\conv3d-00005-0.12532-0.96866-0.27041-0.93000.h5")

        conv3dModel = models.load_model(
            r"D:\PyCharm\Projects\GestureRecognition\conv3d\conv3d_init_2023-01-0101_51_17.081632\conv3d-00017-0.07573-0.97761-0.68541-0.88000.h5")

        # Logic for consolidating the results of the two models
        noOfEqual = 0
        totalNoOfIters = (valOFGen.numVideos / valOFGen.batch_size)
        iter = 0
        for ((convBatchData, convBatchLabels), (ofBatchData, ofBatchLabels)) in zip(convValGen, valOFGen):
            # Conv3D model's pred
            conv3dPreds = conv3dModel.predict(convBatchData)

            # OF model's pred
            ofModelPreds = ofModel.predict(ofBatchData)

            # Averaging the two
            final_predProbas = (ofModelPreds + conv3dPreds) / 2

            # Obtaining predictions with argmax
            final_pred = np.argmax(final_predProbas, axis=1)  # Max across each row

            # Obtain acutal label
            actualLabel = np.argmax(convBatchLabels, axis=1)

            # For calculating accuracy:
            noOfEqual += sum([1 for pred, act in zip(final_pred, actualLabel) if pred == act])
            iter += 1
            if iter >= totalNoOfIters: break

        # Print accuracy.
        accuracy = noOfEqual / valOFGen.numVideos
        print("Accuracy: ", accuracy)


if __name__ == "__main__":
    driver()
