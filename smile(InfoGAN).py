from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.models import load_model
from matplotlib import pyplot
from numpy import savez_compressed
from tensorflow.keras.utils import to_categorical
from numpy import hstack
from numpy import load
from numpy import mean
from numpy import vstack
from numpy import expand_dims

class First_Look:
    def __init__(self,latent_dim, n_cat, n_samples,add):
        self.latent_dim = latent_dim
        self.n_cat = n_cat
        self.n_samples = n_samples
        self.add = add
        #self.generated_points = None
    def generate_latent_points(self):
        x_latent = randn(self.latent_dim * self.n_samples).reshape(self.n_samples, self.latent_dim)
        onehot_codes = to_categorical(randint(0,self.n_cat,self.n_samples), num_classes=self.n_cat)
        return [hstack((x_latent, onehot_codes)), onehot_codes]
    def plot_prelook(self,examples,n):
        for i in range(n * n):
            pyplot.subplot(n, n, 1 + i)
            pyplot.axis('off')
            pyplot.imshow(examples[i, :, :])
            pyplot.savefig('generated_faces.png')
        pyplot.close()
    def execu(self):
        model = load_model(self.add)#'/content/generator_model_100 (1).h5'
        [latent_points,_] = self.generate_latent_points()#90,10,100
        savez_compressed('latent_points.npz', latent_points)
        X = (model.predict(latent_points) + 1) / 2.0
        self.plot_prelook(X, 10)

class Target_Face:
    def __init__(self,add,smiling_woman_ix,neutral_woman_ix,neutral_man_ix):
        self.smiling_woman_ix = smiling_woman_ix
        self.neutral_woman_ix = neutral_woman_ix
        self.neutral_man_ix = neutral_man_ix
        self.S_woman = None
        self.N_woman = None
        self.S_man = None
        self.model  =None
        self.add = add
    def plot_generated(self,examples, rows, cols):
        i = 0
        while i<rows*cols:
            pyplot.subplot(rows, cols, 1 + i)
            pyplot.axis('off')
            pyplot.imshow(examples[i, :, :])
            pyplot.savefig('picked_faces.png')
            i = i + 1
        pyplot.close()
    def average_points(self,points, ix):
        zero_ix = [i-1 for i in ix]
        vectors = points[zero_ix]
        avg_vector = mean(vectors, axis=0)
        all_vectors = vstack((vectors, avg_vector))
        return all_vectors
    def show_ave(self):
        self.model = load_model(self.add)
        self.smiling_woman_ix = [2, 64, 100]
        self.neutral_woman_ix = [22, 78, 10]
        self.neutral_man_ix = [1, 36, 99]
        #data = load('latent_points.npz')
        data = load('latent_points.npz')
        points = data['arr_0']
        self.smiling_woman = self.average_points(points,self.smiling_woman_ix)
        self.neutral_woman = self.average_points(points, self.neutral_woman_ix)
        self.neutral_man = self.average_points(points, self.neutral_man_ix)
        all_vectors = vstack((self.smiling_woman, self.neutral_woman, self.neutral_man))
        #images = model.predict(all_vectors)
        images = (self.model.predict(all_vectors) + 1) / 2.0
        self.plot_generated(images, 3, 4)
    def target_face(self):
        result_vector = self.smiling_woman[-1] - self.neutral_woman[-1] + self.neutral_man[-1]
        result_vector = expand_dims(result_vector, 0)
        result_image = (self.model.predict(result_vector) + 1) / 2.0
        pyplot.imshow(result_image[0])
        pyplot.savefig('target_faces.png')
        pyplot.show()
        pyplot.close()

if __name__ == "__main__":
    mod = int(input("Choose a model(0:directly generate target faces  1:Rerun the faces):"))
    if mod == 1:
        first_look_obj = First_Look(90,10,100,"smile\generator_model_100.h5")
        first_look_obj.generate_latent_points()
        first_look_obj.execu()
    smiling_woman_ix = eval(input("Enter smiling_woman_ix([x,y,z]):"))
    neutral_woman_ix = eval(input("Enter neutral_woman_ix([x,y,z]):"))
    neutral_man_ix = eval(input("Enter neutral_man_ix([x,y,z]):"))
    Target_Face_obj = Target_Face("smile\generator_model_100.h5",smiling_woman_ix,neutral_woman_ix,neutral_man_ix,)#smiling_woman_ix,neutral_woman_ix,neutral_man_ix,
    Target_Face_obj.show_ave()
    Target_Face_obj.target_face()
