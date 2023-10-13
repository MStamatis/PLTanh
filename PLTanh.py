"""
Python (Keras) implemenation of PLTanh activation function used in 
"Parametric Leaky Tanh: A New Hybrid Activation Function for Deep Learning",
https://arxiv.org/ftp/arxiv/papers/2310/2310.07720.pdf
"""
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, Lambda
from tensorflow.keras.utils import get_custom_objects
def SigmoReLU(x):
  """
  The gradients are automatically calculated on TF2
  """
  alpha = 0.01
  return K.maximum(tf.keras.activations.tanh(x), alpha*K.abs(x))
  
#usage between convolution layers
get_custom_objects().update({'PLTanh':
tf.keras.layers.Activation(PLTanh)})

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='PLTanh', input_shape(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(10, activation='softmax')
])
