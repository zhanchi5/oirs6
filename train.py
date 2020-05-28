import matplotlib

matplotlib.use("Agg")

from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
import cv2
from model import Autoencoder
from keras.callbacks import TensorBoard


TRAIN_DATA = 'data/train'
MODEL = 'output/model.h5'
PLOT_PATH = 'plot.png'

EPOCHS = 100
BS = 10

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

# INIT
print("[INFO] loading images...")
data = []
image_paths = list(paths.list_images(TRAIN_DATA))


for image_path in image_paths:
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
    data.append(image)

# learn (75%) test (25%)
trainX, testX = train_test_split(data, test_size=0.2)

trainX = np.asarray(trainX).astype("float32") / 255.0
testX = np.asarray(testX).astype("float32") / 255.0

# Noises
trainNoise = np.random.normal(loc=0.5, scale=0.5, size=trainX.shape)
testNoise = np.random.normal(loc=0.5, scale=0.5, size=testX.shape)
trainXNoisy = np.clip(trainX + trainNoise, 0, 1)
testXNoisy = np.clip(testX + testNoise, 0, 1)

print("[INFO] building autoencoder...")


autoencoder = Autoencoder().build(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
autoencoder.compile(loss="binary_crossentropy", optimizer='adadelta')

H = autoencoder.fit(
    trainXNoisy, trainX,
    validation_data=(testXNoisy, testX),
    epochs=EPOCHS,
    batch_size=BS,
    shuffle=True,
    callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

autoencoder.save(MODEL)
