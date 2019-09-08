


# In[ ]: Speech Stutttering Detection and Removal Project by Shaswat Rajput


from __future__ import division, absolute_import
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')
#%matplotlib inline

import librosa
import librosa.display
import IPython.display

import os
import sys
import re
import shutil
import datetime
import logging
import colorlog
import progressbar
import wave
import pandas as pd
from scipy.io.wavfile import read

import tensorflow as tf
from sklearn.model_selection import train_test_split

#import plotly.plotly as py
#import plotly
#plotly.tools.set_credentials_file(username='#####', api_key='######')
# Setting up progressbar and logger
progressbar.streams.wrap_stderr()
logger = colorlog.getLogger("SSR")
handler = logging.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter('%(log_color)s%(levelname)-8s| %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ## Data Preparation

# In[ ]:


class FeatureExtraction:
    def __init__(self, n_mels=128):
        self.n_mels = n_mels
        self.y = None
        self.sr = None
        self.S = None
        self.log_S = None
        self.mfcc = None
        self.delta_mfcc = None
        self.delta2_mfcc = None
        self.M = None
        self.rmse = None
    
    def loadFile(self, filename):
        self.y, self.sr = librosa.load(filename)
        logger.debug('File loaded: %s', filename)
    
    def load_y_sr(self, y, sr):
        self.y = y
        self.sr = sr
    
    def melspectrogram(self):
        self.S = librosa.feature.melspectrogram(self.y, sr=self.sr, n_mels=self.n_mels)
        self.log_S = librosa.amplitude_to_db(self.S, ref=np.max)
    
    def plotmelspectrogram(self):
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(self.log_S, sr=self.sr, x_axis='time', y_axis='mel')
        plt.title('mel Power Spectrogram')
        plt.colorbar(format='%+02.0f dB')
        plt.plot()
        plt.tight_layout()
    
    def extractmfcc(self, n_mfcc=13):
        self.mfcc = librosa.feature.mfcc(S=self.log_S, n_mfcc=n_mfcc)
        self.delta_mfcc = librosa.feature.delta(self.mfcc,mode='nearest')
        self.delta2_mfcc = librosa.feature.delta(self.mfcc, order=2,mode='nearest')
        self.M = np.vstack([self.mfcc, self.delta_mfcc, self.delta2_mfcc])
    
    
    def plotmfcc(self):
        plt.figure(figsize=(12, 6))
        plt.subplot(3, 1, 1)
        librosa.display.specshow(self.mfcc)
        plt.ylabel('MFCC')
        plt.plot()
        plt.colorbar()
        
        plt.subplot(3, 1, 2)
        librosa.display.specshow(self.delta_mfcc)
        plt.ylabel('MFCC-$\Delta$')
        plt.plot()
        plt.colorbar()
        
        plt.subplot(3, 1, 3)
        librosa.display.specshow(self.delta2_mfcc, sr=self.sr, x_axis='time')
        plt.ylabel('MFCC-$\Delta^2$')
        plt.colorbar()
        plt.plot()
        
        plt.tight_layout()
    
    def extractrmse(self):
        self.rmse = librosa.feature.rmse(y=self.y)


# In[ ]:


class Dataset:
    def __init__(self, datasetDir, datasetLabelFilename, datasetArrayFilename):
        self.n_features = 80
        logger.info("Number of features: %s", self.n_features)
        self.X = np.empty(shape=(0, self.n_features))
        self.Y = np.empty(shape=(0, 2))
        
        self.datasetArrayFilename = datasetArrayFilename
        logger.debug("Dataset array filename: %s", self.datasetArrayFilename)
        
        if os.path.isfile(self.datasetArrayFilename):    
            self.__readFromFile()
        else:
            self.datasetDir = datasetDir
            logger.debug("Dataset Directory: %s", self.datasetDir)

            self.datasetLabelFilename = datasetLabelFilename
            logger.debug("Dataset labels filename: %s", self.datasetLabelFilename)

            if not os.path.isdir(self.datasetDir) or not os.path.isfile(self.datasetLabelFilename):
                logger.info("%s or %s does not exists", self.datasetDir, self.datasetLabelFilename)
                self.__buildDatasetAndLabels('wav/release1')
                
            self.__build()
            self.__writeToFile()
    
    def __build(self):
        logger.info("Building dataset from directory: %s", self.datasetDir)
        num_lines = sum(1 for line in open(self.datasetLabelFilename, 'r'))
        with open(self.datasetLabelFilename, 'r') as datasetLabelFile:
            filesProcessed=0
            pbar = progressbar.ProgressBar(redirect_stdout=True)
            for line in pbar(datasetLabelFile, max_value=num_lines):
                lineSplit = line.strip().split(' ')
                audiofilename = lineSplit[0]
                label = lineSplit[1]
                try:
                    features = FeatureExtraction()
                    features.loadFile(os.path.join(self.datasetDir, audiofilename))
                    features.melspectrogram()
                    features.extractmfcc()
                    features.extractrmse()
                except ValueError:
                    logger.warning("Error in extracting features from file %s", audiofilename)
                    continue
                
                featureVector = []
                for feature in features.mfcc:
                    featureVector.append(np.mean(feature))
                    featureVector.append(np.var(feature))
                
                for feature in features.delta_mfcc:
                    featureVector.append(np.mean(feature))
                    featureVector.append(np.var(feature))
                
                for feature in features.delta2_mfcc:
                    featureVector.append(np.mean(feature))
                    featureVector.append(np.var(feature))
                
                featureVector.append(np.mean(features.rmse))
                featureVector.append(np.var(features.rmse))
                
                self.X = np.vstack((self.X, [featureVector]))
                
                if label == "STUTTER":
                    self.Y = np.vstack((self.Y, [0, 1]))
                elif label == "NORMAL":
                    self.Y = np.vstack((self.Y, [1, 0]))
                else:
                    logger.error("Unexpected label: %s", label)
                    sys.exit()
                
                filesProcessed += 1            
            
            logger.info("Total files processed: %d", filesProcessed)
    
    def __buildDatasetAndLabels(self, audioAndChaFilesDirectory):
        logger.info("Rebuilding the dataset directory and labels")
        if os.path.isdir(self.datasetDir):
            shutil.rmtree(self.datasetDir)
        os.makedirs(self.datasetDir)
        
        labelFile = open(self.datasetLabelFilename, 'w')
        
        splitDuration = 300 # milliseconds
        pbar = progressbar.ProgressBar(redirect_stdout=True)
        for chaFileName in pbar(os.listdir(audioAndChaFilesDirectory)):
            if chaFileName.endswith(".cha"):
                subject = chaFileName.split('.')[0]
                wavFileName = subject + ".wav"
                y, sr = librosa.load(os.path.join(audioAndChaFilesDirectory, wavFileName))

                logger.debug("Parsing file: %s", chaFileName)

                with open(os.path.join(audioAndChaFilesDirectory, chaFileName), 'r') as chaFile:
                    sndFound = False
                    phoFound = False
                    startTime = -1
                    endTime = -1
                    label = None
                    for line in chaFile:
                        if not sndFound:
                            if re.search(r"%snd:", line):
                                lineSplit = line.split("_")
                                startTime = int(lineSplit[-2])
                                endTime = lineSplit[-1]
                                endTime = int(re.sub(r"\u0015\n", '', endTime))
                                sndFound = True
                        else:
                            if re.search(r"%pho:", line):
                                if re.search(r'[A-Z]', line):
                                    label = "STUTTER"
                                else:
                                    label = "NORMAL"
                                phoFound = True
                        if sndFound and phoFound:
                            n_splits = int(np.round((endTime - startTime) / splitDuration))
                            
                            startingSample = int(startTime * sr / 1000)
                            for i in range(1, n_splits):
                                endingSample = int(startingSample + (splitDuration * sr / 1000))
                                audiofilename = subject + ":" + str(startTime) + ":" + str(int(startTime) + splitDuration) + ".wav"
                                labelFile.write(audiofilename + " " + label + "\n")
                                audio = y[startingSample:endingSample]
                                librosa.output.write_wav(os.path.join(self.datasetDir, audiofilename), audio, sr)
                                
                                startingSample = endingSample
                                startTime = int(startTime) + splitDuration
                            
                            endingSample = int(endTime * sr / 1000)
                            audiofilename = subject + ":" + str(startTime) + ":" + str(endTime) + ".wav"
                            labelFile.write(audiofilename + " " + label + "\n")
                            audio = y[startingSample:endingSample]
                            librosa.output.write_wav(os.path.join(self.datasetDir, audiofilename), audio, sr)
                            
                            
                            sndFound = False
                            phoFound = False
                            startTime = -1
                            endTime = -1
                            label = None

        labelFile.close()
    
    def __writeToFile(self, filename=None):
        if filename == None:
            filename = self.datasetArrayFilename
            
        if os.path.exists(filename):
            os.remove(filename)
        np.savetxt(filename, np.hstack((self.X, self.Y)))
        logger.info("Array stored in file %s", filename)
    
    def __readFromFile(self, filename=None):
        if filename == None:
            filename = self.datasetArrayFilename
            
        if not os.path.isfile(filename):
            logger.error("%s does not exists or is not a file", filename)
            sys.exit()
        matrix = np.loadtxt(filename)
        self.X = matrix[:, 0:self.n_features]
        self.Y = matrix[:, self.n_features:]
        logger.info("Array read from file %s", filename)


# ## Tensorflow binary classification

# In[ ]:


class NeuralNetwork:
    def __init__(self, X_train=None, Y_train=None, X_test=None, Y_test=None):
        # Data
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        
        # Learning Parameters
        self.learning_rate = 0.001
        self.training_epochs = 1200
        self.batch_size = 100
        self.display_step = 100

        # Model Parameters
        self.n_hidden = [10, 10, 10]
        self.hiddenLayers = len(self.n_hidden)
        self.n_input = 80
        self.n_classes = 2

        logger.debug("Neural network of depth %d", self.hiddenLayers)
        for i in range(self.hiddenLayers):
            logger.debug("Depth of layer %d is %d", (i + 1), self.n_hidden[i])

        self.x = tf.placeholder("float", [None, self.n_input])
        self.y = tf.placeholder("float", [None, self.n_classes])
        self.layer = None
        self.weights = None
        self.biases = None
        # Model
        self.model = self.__network(self.x)
        self.save_path = None

        # Loss function and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.model, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # Initialize the variables
        self.init = tf.global_variables_initializer()
    
    def setTrainData(self, X, Y):
        self.X_train = X
        self.Y_train = Y
        
    def setTestData(self, X, Y):
        self.X_test = X
        self.Y_test = Y
        
    def __network(self, x):
        self.layer = []
        self.weights = []
        self.biases = []

        for n_layer in range(self.hiddenLayers):
            if n_layer == 0:
                self.weights.append(tf.Variable(tf.random_normal([self.n_input, self.n_hidden[n_layer]])))
                self.biases.append(tf.Variable(tf.random_normal([self.n_hidden[n_layer]])))
                self.layer.append(tf.nn.relu(tf.add(tf.matmul(x, self.weights[n_layer]), self.biases[n_layer])))
            else:
                self.weights.append(tf.Variable(tf.random_normal([self.n_hidden[n_layer - 1], self.n_hidden[n_layer]])))
                self.biases.append(tf.Variable(tf.random_normal([self.n_hidden[n_layer]])))
                self.layer.append(tf.nn.relu(tf.add(tf.matmul(self.layer[n_layer - 1], self.weights[n_layer]), self.biases[n_layer])))


        # Output layer
        self.weights.append(tf.Variable(tf.random_normal([self.n_hidden[self.hiddenLayers - 1], self.n_classes])))
        self.biases.append(tf.Variable(tf.random_normal([self.n_classes])))
        self.layer.append(tf.matmul(self.layer[self.hiddenLayers - 1], self.weights[self.hiddenLayers]) + self.biases[self.hiddenLayers])

        return self.layer[self.hiddenLayers]
    
    def train(self):
        logger.info("Training the neural network")
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(self.init)
            pbarWidgets = [
                progressbar.Percentage(),
                ' (',
                progressbar.SimpleProgress(),
                ') ',
                progressbar.Bar(),
                ' ',
                progressbar.Timer(),
                ' ',
                progressbar.ETA(),
                ' ',
                progressbar.DynamicMessage('Cost'),
            ]
            with progressbar.ProgressBar(max_value=self.training_epochs, redirect_stdout=True, widgets=pbarWidgets) as pbar:
                for epoch in range(self.training_epochs):
                    avg_cost = 0
                    total_batch = int(len(self.X_train) / self.batch_size)
                    X_batches = np.array_split(self.X_train, total_batch)
                    Y_batches = np.array_split(self.Y_train, total_batch)

                    for i in range(total_batch):
                        batch_x, batch_y = X_batches[i], Y_batches[i]
                        # Run optimization op (backprop) and cost op (to get loss value)
                        _, c = sess.run([self.optimizer, self.cost], feed_dict={self.x: batch_x, self.y: batch_y})

                        # Compute average loss
                        avg_cost += c / total_batch

                    pbar.update(epoch + 1, Cost=avg_cost)
                
            logger.info("Optimization Finished!")

            evalAccuracy = self.__getAccuracy()
            
            global result 
            result = tf.argmax(self.model, 1).eval({self.x: self.X_test, self.y: self.Y_test})
            
            tfSessionsDir = "tfSessions"
            if not os.path.isdir(tfSessionsDir):
                os.makedirs(tfSessionsDir)
            timestamp = '{:%Y-%m-%d-%H:%M:%S}'.format(datetime.datetime.now()) + '-' + str(evalAccuracy)
            os.makedirs(os.path.join(tfSessionsDir, timestamp))
            modelfilename =  os.path.join(os.path.join(tfSessionsDir, timestamp), 'session.ckpt')
            self.save_path = saver.save(sess, modelfilename)
            
            with open(os.path.join(os.path.join(tfSessionsDir, timestamp), 'details.txt'), 'w') as details:
                details.write("learning_rate = " + str(self.learning_rate) + "\n")
                details.write("training_epochs = " + str(self.training_epochs) + "\n")
                details.write("batch_size = " + str(self.batch_size) + "\n")
                details.write("display_step = " + str(self.display_step) + "\n")
                details.write("n_hidden = " + str(self.n_hidden) + "\n")
                details.write("hiddenLayers = " + str(self.hiddenLayers) + "\n")
                details.write("n_input = " + str(self.n_input) + "\n")
                details.write("n_classes = " + str(self.n_classes) + "\n")
                
            logger.info("Model saved in file: %s" % self.save_path)
    
    def getModelPath(self):
        return self.save_path
        
    def __getAccuracy(self):
        # Test model
        correct_prediction = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        evalAccuracy = accuracy.eval({self.x: self.X_test, self.y: self.Y_test})
        logger.info("Accuracy: %f", evalAccuracy)
        return evalAccuracy
        
    def loadAndClassify(self, filename, X):            
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, filename)
            prediction_model = tf.argmax(self.model, 1)
            return prediction_model.eval({self.x: X})
            


# ## Using the NN model for classification

# In[ ]:


class AudioCorrection():
    def __init__(self, audiofile, tfSessionFile, segmentLength=300, segmentHop=100, n_features=80, correctionsDir='corrections'):
        self.tfSessionFile = tfSessionFile
        self.segmentLength = segmentLength
        self.segmentHop = segmentHop
        self.n_features = n_features
        self.correctionsDir = correctionsDir
        self.samplesPerSegment = None
        self.samplesToSkipPerHop = None
        self.upperLimit = None
        self.inputFilename = None
        self.y = None
        self.sr = None
        self.target_sr = 16000
        NORMAL = 0
        STUTTER = 1
        self.speech = {NORMAL: [], STUTTER: []}
        self.smoothingSamples = 1000
        self.__loadfile(audiofile)
    
    def __loadfile(self, inputFilename):
        if not os.path.isfile(inputFilename):
            logger.error("%s does not exists or is not a file", inputFilename)
            sys.exit()
        self.inputFilename = inputFilename
        logger.info("Loading file %s", self.inputFilename)
        self.y, self.sr = librosa.load(self.inputFilename)
        self.samplesPerSegment = int(self.segmentLength * self.sr / 1000)
        self.samplesToSkipPerHop = int(self.segmentHop * self.sr / 1000)
        self.upperLimit = len(self.y) - self.samplesPerSegment

    def process(self):
        logger.info("Attempting to correct %s", self.inputFilename)
        X = np.empty(shape=(0, self.n_features))
        durations = np.empty(shape=(0, 2))

        pbar = progressbar.ProgressBar()
        start = 0
        end = 0
        for start in pbar(range(0, self.upperLimit, self.samplesToSkipPerHop)):
            end = start + self.samplesPerSegment
            audio = self.y[start:end]

            featureVector = self.__getFeatureVector(audio, self.sr)
            if featureVector != None:
                X = np.vstack((X, [featureVector]))
                durations = np.vstack((durations, [start, end]))
        
        audio = self.y[end:]
        featureVector = self.__getFeatureVector(audio, self.sr)
        if featureVector != None:
            X = np.vstack((X, [featureVector]))
            durations = np.vstack((durations, [end, self.upperLimit + self.samplesPerSegment]))
        logger.debug("Finished extracting features")

        tf.reset_default_graph()
        nn = NeuralNetwork()
        classificationResult = nn.loadAndClassify(self.tfSessionFile, X)
        logger.debug("Finished classification of segments")
        
        currentSegment = {'type': classificationResult[0], 'start': durations[0][0], 'end': durations[0][1]}
        for (label, [start, end]) in zip(classificationResult[1:], durations[1:]):
            if currentSegment['type'] == label:
                currentSegment['end'] = end
            else:
                self.speech[currentSegment['type']].append((currentSegment['start'], currentSegment['end']))
                currentSegment['type'] = label
                currentSegment['start'] = start
                currentSegment['end'] = end
    
    def __getFeatureVector(self, y, sr):
        try:
            features = FeatureExtraction()
            features.load_y_sr(y, sr)
            features.melspectrogram()
            features.extractmfcc()
            features.extractrmse()
        except ValueError:
            logger.warning("Error extracting features")
            return None

        featureVector = []
        for feature in features.mfcc:
            featureVector.append(np.mean(feature))
            featureVector.append(np.var(feature))

        for feature in features.delta_mfcc:
            featureVector.append(np.mean(feature))
            featureVector.append(np.var(feature))

        for feature in features.delta2_mfcc:
            featureVector.append(np.mean(feature))
            featureVector.append(np.var(feature))

        featureVector.append(np.mean(features.rmse))
        featureVector.append(np.var(features.rmse))
        
        return featureVector
    
    def saveCorrectedAudio(self):
        NORMAL = 0
        STUTTER = 1
        if not os.path.isdir(self.correctionsDir):
            os.makedirs(self.correctionsDir)
        outputFilenamePrefix = os.path.join(self.correctionsDir, os.path.splitext(os.path.basename(self.inputFilename))[0])
        
        normalSpeech = np.ndarray(shape=(1, 0))
        (start, end) = self.speech[NORMAL][0]
        normalSpeech = np.append(normalSpeech, self.y[int(start):int(end)])
        for (start, end) in self.speech[NORMAL][1:]:
            # Smoothing
            previousSample = normalSpeech[-1]
            nextSample = self.y[int(start)]
            if nextSample > previousSample:
                low, high = previousSample, nextSample
            else:
                low, high = nextSample, previousSample
            
            step = (high - low) / self.smoothingSamples
            
            normalSpeech = np.append(normalSpeech, np.arange(low, high, step))
            normalSpeech = np.append(normalSpeech, self.y[int(start):int(end)])

        stutteredSpeech = np.ndarray(shape=(1, 0))
        for (start, end) in self.speech[STUTTER]:
            stutteredSpeech = np.append(stutteredSpeech, self.y[int(start):int(end)])

        # Resampling the audio
        logger.debug("Resampling corrected audio from %d to %d", self.sr, self.target_sr)
        resampledNormalSpeech = librosa.resample(normalSpeech, self.sr, self.target_sr)
        resampledStutteredSpeech = librosa.resample(stutteredSpeech, self.sr, self.target_sr)
        librosa.output.write_wav(outputFilenamePrefix + "-corrected.wav", normalSpeech, self.sr)
        librosa.output.write_wav(outputFilenamePrefix + "-stuttered.wav", stutteredSpeech, self.sr)
        logger.info("Corrected audio saved as %s", outputFilenamePrefix + "-corrected.wav")


# In[ ]:


def run(train=False, correct=False):
    if train:
        dataset = Dataset('dataset', 'datasetLabels.txt', 'datasetArray80.gz')
        X_train, X_test, Y_train, Y_test = train_test_split(dataset.X, dataset.Y)

        tf.reset_default_graph()
        nn = NeuralNetwork(X_train, Y_train, X_test, Y_test)
        nn.train()

    if correct:
        audiofile = 'M_0219_11y2m_1.wav'
        if train:
            tfSessionFile = nn.getModelPath()
        else:
            tfSessionFile = 'tfSessions/2018-10-08-12:57:33-0.8590971/session.ckpt'

        correction = AudioCorrection(audiofile, tfSessionFile)
        correction.process()
        correction.saveCorrectedAudio()


# In[ ]:
input_data = read("M_0219_11y2m_1.wav")
audio = input_data[1]
plt.figure(figsize=(20,7))
plt.plot(audio[0:])
plt.ylabel("Amplitude")
plt.xlabel("Time")
# set the title  
plt.title("Sample Wav")
# display the plot
plt.show()

#py.iplot(audio[0:])

if __name__ == "__main__":
    run(False, True)

input_data = read("corrections/M_0219_11y2m_1-corrected.wav")
audio = input_data[1]
plt.figure(figsize=(20,7))
plt.plot(audio[0:])
plt.ylabel("Amplitude")
plt.xlabel("Time")
    
    
    
    
    
