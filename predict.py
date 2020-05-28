import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from imutils import paths

MODEL = 'output/model.h5'
RESULT = 'data/result'
TEST_IMAGES = 'data/test'

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

model = load_model(MODEL)
test_images = list(paths.list_images(TEST_IMAGES))

for idx, test_image in enumerate(test_images):
    image = cv2.imread(test_image)
    image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))

    image = image.astype("float") / 255.0
    image = np.array([image])

    testNoise = np.random.normal(loc=0.1, scale=0.5, size=image.shape)
    image = np.clip(image + testNoise, 0, 1)

    preds = model.predict(image)
    original = (image * 255).astype("uint8")
    recon = (preds * 255).astype("uint8")

    output = np.hstack([original[0], recon[0]])
    file_path = os.path.join(RESULT, f'{idx}.png')
    cv2.imwrite(file_path, output)