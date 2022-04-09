# # Create your views here.
# import zipfile, os, pickle
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# import os

# import numpy as np
# from tensorflow.keras.applications.vgg19 import VGG19
# from sklearn.metrics.pairwise import rbf_kernel
# # Create your views here.from tensorflow.keras.applications.vgg19 import VGG19
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from tensorflow.keras.applications.vgg19 import preprocess_input
# from keras.models import Model
# import matplotlib.pyplot as plt
# import numpy as np
# import PIL

# class ImageSearch:
#     model = None
#     groups = None
#     kmeans = None
#     features = None
#     cluster = None
#     pca = None
#     def __init__(self) -> None:
#         try:
#             with zipfile.ZipFile(os.path.join("Data", "archive.zip"), 'r') as zip_ref:
#                 zip_ref.extractall("Data")
#         except:
#             pass
#         #extract features from all photos in library
#         self.model = VGG19(weights='imagenet')
#         #remove classification layer
#         self.model.layers.pop()
#         self.model = Model(inputs = self.model.inputs, outputs = self.model.layers[-2].output)

#         with open(os.path.join("Data", "cluster.pkl"), "rb") as input_file:
#             self.groups = pickle.load(input_file)
#         with open(os.path.join("Data", "features.pkl"), "rb") as input_file:
#             self.features = pickle.load(input_file)
#         with open(os.path.join("Data", "kmeans.pkl"), "rb") as input_file:
#             self.kmeans = pickle.load(input_file)
#         with open(os.path.join("Data", "pca.pkl"), "rb") as input_file:
#             self.pca = pickle.load(input_file)

        
#     def extract_features(self, filename):
# 	    # load the photo
#         image = load_img(filename, target_size=(224, 224))
# 	    # convert the image pixels to a numpy array
#         image = img_to_array(image)
# 	    # reshape data for the model
#         image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# 	    # prepare the image for the VGG model
#         image = preprocess_input(image)
# 	    # get features

#         feature = self.model.predict(image, verbose=0)   
#         return feature

#     def search_images(self, img, n=10):
#         feat = self.extract_features(img)
#         dimn = self.pca.transform(feat)
#         pred = self.kmeans.predict(dimn)
#         nfet = np.asarray(feat, dtype=np.float32)
#         lis = []
#         # print(groups[pred[0]])
#         for img in self.groups[pred[0]]:
#             t = (rbf_kernel(nfet, self.features[img]), img)
#             lis.append(t)
#         lis.sort(reverse=True)
#         return lis[:min(n, len(lis))]

#     def search_images_entire(self, img, n=10):
#         feat = self.extract_features(img)
#         dimn = self.pca.transform(feat)
#         pred = self.kmeans.predict(dimn)
#         nfet = np.asarray(feat, dtype=np.float32)
#         lis = []
#         # print(groups[pred[0]])
#         for i in range(50):
#           for img in self.groups[i]:
#               t = (rbf_kernel(nfet, self.features[img]), img)
#               lis.append(t)
#         lis.sort(reverse=True)
#         return lis[:min(n, len(lis))]
    
#     def show_images(self, images, cols = 1, titles = None):
#         """Display a list of images in a single figure with matplotlib.

#         Parameters
#         ---------
#         images: List of np.arrays compatible with plt.imshow.

#         cols (Default = 1): Number of columns in figure (number of rows is 
#                             set to np.ceil(n_images/float(cols))).

#         titles: List of titles corresponding to each image. Must have
#                 the same length as titles.
#         """
#         print(type(images))
#         n_images = len(images)
#         # if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
#         fig = plt.figure()
#         for n, (score,image) in enumerate(images):
#             a = fig.add_subplot(cols, int(np.ceil(n_images/float(cols))), n + 1)
#             if image.ndim == 2:
#                 plt.gray()
#             test = PIL.Image.open(os.path.join("Data", "Images", image))
#             plt.imshow(test)
#             a.set_title(f"similarity score : {score[0][0]}")
#         fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
#         plt.show()

