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


class Generator(keras.utils.Sequence):
    # Member variables

    def __init__(self):
        pass

    def initialize(self):
        pass

    def getBatchData(self, batch, currBatchSize):
        pass

    def getNumBatches(self):
        pass

    # Since we are inheriting from keras.utils.Sequence, we need to implement getitem, len
    def __getitem__(self, batchIdx):
        pass

    def __len__(self):
        pass


## ------------------------ Generator for CNN followed by RNN ----------------------------- ##
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
        # Try other feature extractors
        elif preTrainedModel == "vgg16":
            pass
        else:
            pass

    def getNumBatches(self):
        return self.numBatches

    def initializeFeatureExtractor(self, preTrainedModel="inceptionv3"):
        np.random.seed(30)
        rn.seed(30)
        tf.random.set_seed(30)
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

    # Keeping this for debugging purposes. Not really needed, was used to test next(generator)
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

    # Keeping this for debugging purposes. Not really needed, was used to test next(generator)
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
def get_conv3d_model(numFeaturesInFirstLayer, numFrames, output_activation="softmax",
                     width=224, height=224, numChannels=3,
                     numClasses=5, numNeuronsInDenseLayer=256):
    # Model
    model = models.Sequential()

    # Convolution layer with `numFeaturesInFirstLayer` features, 3x3 filter and a relu activation with 2x2 pooling
    model.add(layers.Conv3D(numFeaturesInFirstLayer, (3, 3, 3), padding='same', activation='relu',
                            input_shape=(numFrames, width, height, numChannels)))
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
    model_name = r"conv3d\conv3d_init" + '_' + str(curr_dt_time).replace(' ', '').replace(':', '_') + '\\'
    currDir = os.getcwd()
    if not os.path.exists(currDir + "\\" + "conv3d"):
        os.mkdir(currDir + "\\" + "conv3d")

    modelPath = currDir + "\\" + model_name
    if not os.path.exists(modelPath):
        os.mkdir(modelPath)

    filepath = modelPath + 'conv3d-{epoch:05d}-{loss:.5f}-{categorical_accuracy:.5f}-{val_loss:.5f}-{val_categorical_accuracy:.5f}.h5'
    # val_loss
    checkpoint = ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=1,
                                 save_best_only=True, save_weights_only=False,
                                 mode='auto', period=1)
    LR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2,
                           min_lr=1e-4)  # write the REducelronplateau code here
    return [checkpoint, LR]
    # return [LR]


## -- Conv3d done -- ##

def driver():
    # Some constants
    frameIdxList = list(range(0, 30))
    width = 224
    height = 224
    numChannels = 3
    nClasses = 5
    output_activation = "softmax"
    batch_size = 10
    num_epochs = 20

    val_path = r"D:\DDownloads\UpGrad\NeuralNetwork\CaseStudy\Project_data\val"
    val_csv = r"D:\DDownloads\UpGrad\NeuralNetwork\CaseStudy\Project_data\val.csv"

    ## Type of model: Modify this.
    modelType = "conv3d"
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

        numFeaturesInFirstLayer = 16
        numNeuronsInDenseLayer = 128
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


if __name__ == "__main__":
    driver()
