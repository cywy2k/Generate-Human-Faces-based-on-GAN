from numpy.random import randn
from numpy.random import randint
from numpy.random import randn
from numpy.random import randint
from numpy import asarray
from numpy import vstack
from numpy import hstack
from numpy import linspace
from matplotlib import pyplot
from keras.models import load_model
from keras.utils import to_categorical


class Interpolation:
    def __init__(self, model_path, latent_dim, n_cat, n_samples, n_steps):
        self.model = load_model(model_path)
        self.latent_dim = latent_dim
        self.n_cat = n_cat
        self.n_samples = n_samples
        self.n_steps = n_steps

    def generate_latent_points_infogan(self):
        x_latent = randn(self.latent_dim * self.n_samples).reshape(self.n_samples, self.latent_dim)
        onehot_codes = to_categorical(randint(0,self.n_cat,self.n_samples), num_classes=self.n_cat)
        return [hstack((x_latent, onehot_codes)), onehot_codes]
 
    def interpolate(self, v1, v2):
        ratios = linspace(0, 1, num=self.n_steps)
        vectors = []
        for r in ratios:
            vectors.append((1.0 - r) * v1 + r * v2)
        return asarray(vectors)

    def plot_results(self,samples,n_plot):
        i = 0
        while i <n_plot **2:
            pyplot.subplot(n_plot, n_plot, 1 + i)
            pyplot.axis('off')
            pyplot.imshow(samples[i, :, :])
            i+=1
        pyplot.show()

    def interpolate_faces(self):
        [points,_] = self.generate_latent_points_infogan()
        results = None
        for i in range(0, self.n_samples, 2):
            interpolated_vector = self.interpolate(points[i], points[i+1])
            prd = self.model.predict(interpolated_vector)
            prd = (prd + 1) / 2.0
            if results is not None:
                results = vstack((results, prd))
            else: results = prd
        return results


if __name__ == "__main__":
    model_path = '/content/generator_model_100.h5'
    mymodel = Interpolation(model_path, latent_dim=90, n_cat=10, n_samples=20, n_steps=10)
    results = mymodel.interpolate_faces()
    mymodel.plot_results(results,10)