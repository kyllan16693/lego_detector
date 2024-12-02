from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, 
    Dropout, 
    BatchNormalization, 
    Activation,
    GlobalAveragePooling2D
)

class EfficientNetV2:
    """
    EfficientNetV2 model wrapper with different variants and configurations
    """
    
    VARIANTS = {
        'b0': applications.EfficientNetV2B0,
        'b1': applications.EfficientNetV2B1,
        'b2': applications.EfficientNetV2B2,
        'b3': applications.EfficientNetV2B3,
        's': applications.EfficientNetV2S,
        'm': applications.EfficientNetV2M,
        'l': applications.EfficientNetV2L
    }
    
    def __init__(self, 
                 variant='b0',
                 dropout_rate=0.5,
                 dense_units=1024,
                 use_extra_dense=True):
        """
        Initialize EfficientNetV2 model configuration
        """
        if variant not in self.VARIANTS:
            raise ValueError(f"Invalid variant. Choose from: {list(self.VARIANTS.keys())}")
            
        self.model_class = self.VARIANTS[variant]
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.use_extra_dense = use_extra_dense
        
    def build(self, image_width, image_height, num_classes, weights='imagenet'):
        """
        Build and return the EfficientNetV2 model
        """
        # Base model
        base_model = self.model_class(
            weights=weights,
            include_top=False,
            input_shape=(image_height, image_width, 3)
        )
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        
        if self.use_extra_dense:
            x = Dense(self.dense_units)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
        x = Dropout(self.dropout_rate)(x)
        
        # Final classification layer
        x = Dense(num_classes)(x)
        outputs = Activation('softmax', name='class_out')(x)
        
        # Create model
        model = Model(inputs=base_model.input, outputs=outputs)
        
        return model
    
    def get_preprocessing(self):
        """Get the preprocessing function for the model variant"""
        return self.model_class.preprocess_input
    
    @staticmethod
    def get_available_variants():
        """Get list of available model variants"""
        return list(EfficientNetV2.VARIANTS.keys()) 