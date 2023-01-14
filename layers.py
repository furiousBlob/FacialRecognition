#custom L1Dist layer module to load our custom model

#import dependencies
import tensorflow as tf
from tensorflow.keras.layers import Layer

#L1Dist layer from jupyter
class L1Dist(Layer):
    
    #init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
    
    # magic happens here - anchor and positive/negative image are compared
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

