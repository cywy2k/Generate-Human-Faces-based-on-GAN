from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from numpy import hstack
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from matplotlib import pyplot

def load_real_samples(path):
	data = load(path)
	data = data['arr_0'].astype('float32')
	data = (data - 127.5) / 127.5  #normalization
	return data

class InfoGAN:
	
	def __init__(self,dataset = None, latent_dim = None, n_catgory = None, n_epochs = None, n_batch = None,n_display=10):
		self.g_model = None
		self.d_model = None
		self.q_model = None
		self.GAN_model = None
		self.dataset = dataset
		self.latent_dim = latent_dim
		self.n_catgory = n_catgory
		self.n_epochs = n_epochs
		self.n_batch = n_batch
		self.n_display = n_display

	def generate_latent_points(self,  n_samples):
		x_latent = randn(self.latent_dim * n_samples)
		x_latent = x_latent.reshape(n_samples, self.latent_dim)
		cat_codes = randint(0,self.n_catgory,n_samples)
		cat_codes = to_categorical(cat_codes, num_classes=self.n_catgory)
		x_input = hstack((x_latent, cat_codes))
		return [x_input, cat_codes]

	def generate_real_samples(self, n_samples):
		ix = randint(0, self.dataset.shape[0], n_samples)
		return dataset[ix], ones((n_samples, 1))

	def generate_fake_samples(self, n_samples):
		x_input,_ = self.generate_latent_points(n_samples)
		input = self.g_model.predict(x_input)
		output = zeros((n_samples, 1))
		return input, output

	def Generator(self):
		model = Sequential()
		init = RandomNormal(stddev=0.02)
		n_nodes = 128 * 5 * 5
		input_size = self.latent_dim + self.n_catgory
		model.add(Dense(n_nodes, input_dim=input_size, kernel_initializer=init))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Reshape((5, 5, 128)))
		model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
		model.add(BatchNormalization(momentum=0.9))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
		model.add(BatchNormalization(momentum=0.9))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
		model.add(BatchNormalization(momentum=0.9))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Conv2D(3, (5,5), activation='tanh', padding='same'))
		self.g_model = model
	
	def Descriminator_Q(self):
		in_put = Input(shape=(80,80,3))
		init = RandomNormal(stddev=0.02)
		d = Conv2D(128, (5,5), padding='same',kernel_initializer =init)(in_put)
		d = LeakyReLU(alpha=0.2)(d)
		d = Conv2D(128, (5,5), strides=(2,2), padding='same',kernel_initializer = init)(d)
		d = LeakyReLU(alpha=0.2)(d)
		d = Conv2D(128, (5,5), strides=(2,2), padding='same',kernel_initializer = init)(d)
		d = LeakyReLU(alpha=0.2)(d)
		d = Conv2D(128, (5,5), strides=(2,2), padding='same',kernel_initializer = init)(d)
		d = LeakyReLU(alpha=0.2)(d)
		d = Conv2D(128, (5,5), strides=(2,2), padding='same', kernel_initializer = init)(d)
		d = LeakyReLU(alpha=0.2)(d)
		d = Flatten()(d)
		model_d = Model(in_put,Dense(1,activation="sigmoid")(d))
		model_d.compile(loss="binary_crossentropy",optimizer=Adam(lr=0.0002, beta_1=0.5),metrics=['accuracy'])

		q = Dense(128)(d) 
		q = BatchNormalization(momentum=0.9)(q)
		q = LeakyReLU(alpha=0.1)(q)
		model_q = Model(in_put, Dense(self.n_catgory, activation='softmax')(q))

		self.d_model = model_d
		self.q_model = model_q	

	def infoGAN_model(self):
		self.d_model.trainable = False
		d_output = self.d_model(self.g_model.output)
		q_output = self.q_model(self.g_model.output)
		model = Model(self.g_model.input,[d_output,q_output])
		model.compile(loss=['binary_crossentropy','categorical_crossentropy'],optimizer=Adam(lr=0.0002, beta_1=0.5))
		self.GAN_model = model

	def save_plot(self, faces, epoch):
		i=0
		n = self.n_display
		faces = (faces + 1) / 2.0
		while i < n**2:
			pyplot.subplot(n, n, 1 + i)
			pyplot.axis('off')
			pyplot.imshow(faces[i])
			i=i+1
		pyplot.savefig('generated_plot_e%03d.png' % (epoch+1))
		pyplot.close()

	def summarize_performance(self,epoch):
		n_samples=(self.n_display)**2
		X_real, y_real = self.generate_real_samples(self.dataset, n_samples)
		x_fake, y_fake = self.generate_fake_samples(self.g_model, n_samples)
		_, acc_real = self.d_model.evaluate(X_real, y_real, verbose=0)
		_, acc_fake = self.d_model.evaluate(x_fake, y_fake, verbose=0)
		print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
		self.save_plot(x_fake, epoch)
		self.g_model.save('generator_model_%03d.h5' % (epoch+1))

	def Train(self):
		bat_per_epo = int(dataset.shape[0] / self.n_batch)
		half_batch = int(self.n_batch / 2)
		i = 0
		while i < self.n_epochs:
			j = 0
			while j < bat_per_epo:
				X_real, y_real = self.generate_real_samples(dataset, half_batch)
				X_fake, y_fake = self.generate_fake_samples(half_batch)	
				X_gan, cat_codes = self.generate_latent_points(self.n_batch)				
				y_gan = ones((self.n_batch,1))
				d_loss_real, _ = self.d_model.train_on_batch(X_real, y_real)	
				d_loss_fake, _ = self.d_model.train_on_batch(X_fake, y_fake)
				_,g_loss,q_loss = self.GAN_model.train_on_batch(X_gan, [y_gan,cat_codes])
				print('>%d, %d/%d, d1=%.3f, d2=%.3f, g=%.3f, q=%.3f' %
					(i+1, j+1, bat_per_epo, d_loss_real, d_loss_fake, g_loss, q_loss))
				j+=1
			if (i+1) % 10 == 0:
				self.summarize_performance(i, self.g_model, self.d_model, self.dataset, self.latent_dim, self.n_catgory)
			i = i+1
				

if __name__ == "__main__":
	dataset = load_real_samples("")
	MyModel = InfoGAN(dataset=dataset, latent_dim=90,n_catgory=10,n_epochs=100,n_batch=64)
	MyModel.Descriminator_Q()
	MyModel.Generator()
	MyModel.infoGAN_model()
	MyModel.Train()
    	