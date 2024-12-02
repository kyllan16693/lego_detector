from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization, Activation

class Model_ConvNeXt(object):
    def create_model(self, image_width, image_height, num_classes):
        base_model = applications.ConvNeXtTiny(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(image_width, image_height, 3)
        )
        
        x = base_model.output
        x = Dropout(0.5)(x)
        x = Dense(num_classes)(x)
        x = BatchNormalization()(x)
        outputs = Activation('softmax', name='class_out')(x)
        
        return Model(inputs=base_model.input, outputs=outputs) 