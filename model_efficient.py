from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization, Activation
from efficientnetv2 import EfficientNetV2

class Model_Efficient(object):
    def create_model(self, image_width, image_height, num_classes):
        # Create EfficientNetV2 instance with custom configuration
        efficient_net = EfficientNetV2(
            variant='b0',  # or 's' for better accuracy
            dropout_rate=0.5,
            dense_units=1024,
            use_extra_dense=True
        )
        
        # Build and return the model
        return efficient_net.build(
            image_width=image_width,
            image_height=image_height,
            num_classes=num_classes,
            weights='imagenet'
        ) 