
from tensorflow.keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Conv2D


encoding_dim = 128

def autoencoder(input_img):
	# Encoder
	#
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)



	conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
	up1 = UpSampling2D((2,2))(conv4)
	conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1) 
	up2 = UpSampling2D((2,2))(conv5) 
	decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) 

	return decoded