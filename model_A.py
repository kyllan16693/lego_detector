from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization, Activation

class Model_A(object):

    def __init__(self) -> None:
        super().__init__()

    def create_model(self, image_width, image_height, num_classes):
        model = applications.ResNet50(weights="imagenet", include_top=False, pooling="avg", input_shape=(image_width, image_height, 3))

        # Custom layer for classification output
        x = model.output
        x = Dropout(0.8)(x)

        # Classification output
        x = Dense(num_classes)(x)
        x = BatchNormalization()(x)
        class_output = Activation('softmax', name='class_out')(x)

        # Define the model with specific outputs
        return Model(inputs=model.input, outputs=class_output)