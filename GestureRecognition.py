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
from tensorflow.keras import layers, models
from keras.models import Sequential, Model
from keras.layers import Dense, GRU, Flatten, TimeDistributed, Flatten, BatchNormalization, Activation, Dropout
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers


class Generator:
    # Member variables

    def __init__(self):
        pass

    def initialize(self):
        pass

    def getBatchData(self, batch, currBatchSize):
        pass

    def generator(self):
        pass


class CNNRNNGenerator(Generator, keras.utils.Sequence):
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


## Model utilities

def get_rnn_model(numFrames=30, numFeatures=2048):
    frame_features_input = keras.Input(shape=(numFrames, numFeatures))
    x = GRU(16, return_sequences=True)(frame_features_input)
    x = GRU(8, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)  # Adding one dense layer
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


if __name__ == "__main__":
    # Some constants
    frameIdxList = list(range(0, 30))
    width = 224
    height = 224
    numChannels = 3
    nClasses = 5
    output_activation = "softmax"
    batch_size = 10
    num_epochs = 10

    # Generators
    train_gen = CNNRNNGenerator(frameIdxList=frameIdxList, batch_size=batch_size)
    train_gen.initializeFeatureExtractor()
    val_path = r"D:\DDownloads\UpGrad\NeuralNetwork\CaseStudy\Project_data\val"
    val_csv = r"D:\DDownloads\UpGrad\NeuralNetwork\CaseStudy\Project_data\val.csv"
    val_gen = CNNRNNGenerator(frameIdxList=frameIdxList, source_path=val_path, batch_size=batch_size, dataCSV=val_csv)
    val_gen.initializeFeatureExtractor()

    # Model
    model = get_rnn_model()
    print(model.summary())
    callbacks_list = create_model_dir()

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
