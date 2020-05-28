from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


class Autoencoder:
	encoding_dim = 49

	def build(self, height, width, channels):
		# Encoder
		input_img = Input(shape=(height, width, channels))

		flat_img = Flatten()(input_img)

		x1 = Dense(self.encoding_dim*3, activation='relu')(flat_img)
		x2 = Dense(self.encoding_dim*2, activation='relu')(x1)
		encoded = Dense(self.encoding_dim, activation='linear')(x2)
		
		# Decoder
		input_encoded = Input(shape=(self.encoding_dim,))
		x1 = Dense(self.encoding_dim*2, activation='relu')(input_encoded)
		x2 = Dense(self.encoding_dim*3, activation='relu')(x1)
		flat_decoded = Dense(height*width*channels, activation='sigmoid')(x2)

		decoded = Reshape((height, width, channels))(flat_decoded)

		# First arg - input layers, second arg - output layers
		encoder = Model(input_img, encoded, name="encoder")
		decoder = Model(input_encoded, decoded, name="decoder")

		autoencoder = Model(input_img, decoder(encoder(input_img)), name="autoencoder")
		return autoencoder