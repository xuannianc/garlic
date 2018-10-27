from keras.utils import plot_model
from keras.applications.vgg16 import VGG16

vgg = VGG16(weights='imagenet')
plot_model(vgg, to_file='vgg16.jpg', show_shapes=True)
