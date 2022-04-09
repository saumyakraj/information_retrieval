# Create your views here.
import zipfile, os, pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

import numpy as np
from tensorflow.keras.applications import inception_resnet_v2
from tensorflow.keras.applications import vgg19

from sklearn.metrics.pairwise import rbf_kernel
# Create your views here.from tensorflow.keras.applications.vgg19 import VGG19
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import PIL

class ImageSearchModels:
    model = None
    groups = None
    kmeans = None
    features = None
    cluster = None
    pca = None
    preprocess_input = None

    image_size = None
    def __init__(self, model, model_preprocess, path, image_size) -> None:
        self.image_size = image_size
        #extract features from all photos in library
        self.model = model(weights='imagenet')
        #remove classification layer
        self.model.layers.pop()
        self.model = Model(inputs = self.model.inputs, outputs = self.model.layers[-2].output)

        with open(os.path.join(path, "cluster.pkl"), "rb") as input_file:
            self.groups = pickle.load(input_file)
        with open(os.path.join(path, "features.pkl"), "rb") as input_file:
            self.features = pickle.load(input_file)
        with open(os.path.join(path, "kmeans.pkl"), "rb") as input_file:
            self.kmeans = pickle.load(input_file)
        with open(os.path.join(path, "pca.pkl"), "rb") as input_file:
            self.pca = pickle.load(input_file)

        self.preprocess_input = model_preprocess

class ImageSearch:
    InceptionResNetV2_model = None
    VGG19_model = None
    model_dict = None

    def __init__(self) -> None:
        try:
            with zipfile.ZipFile(os.path.join("Data", "archive.zip"), 'r') as zip_ref:
                zip_ref.extractall("Data")
        except:
            pass
        self.InceptionResNetV2_model = ImageSearchModels(inception_resnet_v2.InceptionResNetV2, inception_resnet_v2.preprocess_input, os.path.join("Data", "InceptionResNetV2"), 299)
        self.VGG19_model = ImageSearchModels(vgg19.VGG19, vgg19.preprocess_input, os.path.join("Data", "vgg19"), 224)
        self.model_dict = {'vgg19' : self.VGG19_model, 'inceptionresnetv2': self.InceptionResNetV2_model}

    def extract_features(self, file, img_model):
        # load the image as a 224x224 array
        img = load_img(file, target_size=(img_model.image_size,img_model.image_size))
        # convert from 'PIL.Image.Image' to numpy array
        img = np.array(img) 
        # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
        reshaped_img = img.reshape(1,img_model.image_size,img_model.image_size,3) 
        # prepare image for model
        imgx = img_model.preprocess_input(reshaped_img)
        # get the feature vector
        features = img_model.model.predict(imgx, use_multiprocessing=True)
        return features

    def search_images(self, img, model_name, n=10):
        img_model = self.model_dict[model_name]
        feat = self.extract_features(img, img_model)
        dimn = img_model.pca.transform(feat)
        pred = img_model.kmeans.predict(dimn)
        nfet = np.asarray(feat, dtype=np.float32)
        lis = []
        # print(groups[pred[0]])
        for img in img_model.groups[pred[0]]:
            t = (rbf_kernel(nfet, img_model.features[img]), img)
            lis.append(t)
        lis.sort(reverse=True)
        return lis[:min(n, len(lis))]


    
    def show_images(self, images, cols = 1, titles = None):
        """Display a list of images in a single figure with matplotlib.

        Parameters
        ---------
        images: List of np.arrays compatible with plt.imshow.

        cols (Default = 1): Number of columns in figure (number of rows is 
                            set to np.ceil(n_images/float(cols))).

        titles: List of titles corresponding to each image. Must have
                the same length as titles.
        """
        print(type(images))
        n_images = len(images)
        # if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
        fig = plt.figure()
        for n, (score,image) in enumerate(images):
            a = fig.add_subplot(cols, int(np.ceil(n_images/float(cols))), n + 1)
            if image.ndim == 2:
                plt.gray()
            test = PIL.Image.open(os.path.join("Data", "Images", image))
            plt.imshow(test)
            a.set_title(f"similarity score : {score[0][0]}")
        fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
        plt.show()

