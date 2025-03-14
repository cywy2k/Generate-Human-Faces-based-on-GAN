from numpy import load
from numpy import zeros
import sys
from numpy import ones
from numpy.random import randn
import os
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from matplotlib import pyplot


def real_samples_process(data):
	data = data.astype('float32')
	result = data/127.5 - 1
	return result

def generate_real_samples(dataset, n_samples):
	ix = randint(0, dataset.shape[0], n_samples)
	return dataset[ix], ones((n_samples, 1))

def generate_fake_samples(g_model, latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples,latent_dim)
    return g_model.predict(x_input), zeros((n_samples, 1))

class DCGAN:
    def __init__(self,dataset = None, latent_dim = None, n_epochs = None, n_batch = None,n_display=10):
        self.g_model = None
        self.d_model = None
        self.GAN_model = None
        self.dataset = dataset
        self.latent_dim = latent_dim
        self.n_epochs = n_epochs
        self.n_batch = n_batch
        self.n_display = n_display

    def Generator(self):
        model = Sequential()
        n_nodes = 128 * 5 * 5
        model.add(Dense(n_nodes, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((5, 5, 128)))
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        model.add(BatchNormalization(momentum = 0.95))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        model.add(BatchNormalization(momentum = 0.95))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        model.add(BatchNormalization(momentum = 0.95))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(3, (5,5), activation='tanh', padding='same'))
        self.g_model = model
    def Descriminator(self,in_shape=(80,80,3)):
        model = Sequential()
        model.add(Conv2D(128, (5,5), padding='same', input_shape=in_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, (5,5), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.d_model = model
    def GAN(self):
        self.d_model.trainable = False
        model = Sequential()
        model.add(self.g_model)
        model.add(self.d_model)
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        self.GAN_model = model
    def save_plot(self, examples, epoch):
        n = self.n_display
        examples = (examples + 1) / 2.0
        i=0
        while i<n**2:
            pyplot.subplot(n, n, 1 + i)
            pyplot.axis('off')
            pyplot.imshow(examples[i])
            i = i + 1
        filename = 'generated_plot_e%03d.png' % (epoch+1)
        pyplot.savefig(filename)
        pyplot.close()
    def summarize_performance(self,epoch):
        n_samples=(self.n_display)**2
        X_real, y_real = generate_real_samples(self.dataset, n_samples)
        _, acc_real = self.d_model.evaluate(X_real, y_real, verbose=0)
        x_fake, y_fake = generate_fake_samples(self.g_model, self.latent_dim, n_samples)
        _, acc_fake = self.d_model.evaluate(x_fake, y_fake, verbose=0)
        print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
        self.save_plot(x_fake, epoch)
        filename = 'generator_model_%03d.h5' % (epoch+1)
        self.g_model.save(filename)
    def Train(self):
        bat_per_epo = int(self.dataset.shape[0] / self.n_batch)
        half_batch = int(self.n_batch / 2)
        """
        for i in range(self.n_epochs):
            for j in range(bat_per_epo):
                X_real, y_real = generate_real_samples(self.dataset, half_batch)
                d_loss1, _ = self.d_model.train_on_batch(X_real, y_real)
                X_fake, y_fake = generate_fake_samples(self.g_model, self.latent_dim, half_batch)
                d_loss2, _ = self.d_model.train_on_batch(X_fake, y_fake)
                temp = randn(self.latent_dim * self.n_batch)
                X_gan = temp.reshape(self.n_batch, self.latent_dim)
                y_gan = ones((self.n_batch, 1))
                g_loss = self.GAN_model.train_on_batch(X_gan, y_gan)
                print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' % (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
            if (i+1) % 10 == 0:
                self.summarize_performance(i)
        """
        i = 0
        while i < self.n_epochs:
            j = 0
            if (i+1) % 10 ==0:
                while j < bat_per_epo:
                    X_real, y_real = generate_real_samples(self.dataset, half_batch)
                    d_loss1, _ = self.d_model.train_on_batch(X_real, y_real)
                    X_fake, y_fake = generate_fake_samples(self.g_model, self.latent_dim, half_batch)
                    d_loss2, _ = self.d_model.train_on_batch(X_fake, y_fake)
                    temp = randn(self.latent_dim * self.n_batch)
                    X_gan = temp.reshape(self.n_batch, self.latent_dim)
                    y_gan = ones((self.n_batch, 1))
                    g_loss = self.GAN_model.train_on_batch(X_gan, y_gan)
                    print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' % (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
                    j = j + 1
                self.summarize_performance(i)
            else:
                while j < bat_per_epo:
                    X_real, y_real = generate_real_samples(self.dataset, half_batch)
                    d_loss1, _ = self.d_model.train_on_batch(X_real, y_real)
                    X_fake, y_fake = generate_fake_samples(self.g_model, self.latent_dim, half_batch)
                    d_loss2, _ = self.d_model.train_on_batch(X_fake, y_fake)
                    temp = randn(self.latent_dim * self.n_batch)
                    X_gan = temp.reshape(self.n_batch, self.latent_dim)
                    y_gan = ones((self.n_batch, 1))
                    g_loss = self.GAN_model.train_on_batch(X_gan, y_gan)
                    print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' % (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
                    j = j + 1
            i = i + 1

            
#http://moss.stanford.edu/results/8/4902409793316/match0.html
#python DCGAN.py 100 100 64 C:\Users\86147\Desktop\4210\DCGAN\training_data.npz
if __name__ == "__main__":
    latent_dim = int(sys.argv[1])
    n_epochs = int(sys.argv[2])
    n_batch = int(sys.argv[3])
    path = sys.argv[4]
    data = load(path)
    dataset = real_samples_process(data['arr_0'])
    MyModel = DCGAN(dataset=dataset,latent_dim=latent_dim,n_epochs=n_epochs,n_batch=n_batch)
    MyModel.Descriminator()
    MyModel.Generator()
    MyModel.GAN()
    MyModel.Train()


