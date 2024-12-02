import argparse
import numpy as np
import os
from datetime import datetime

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

from keras import optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from keras.losses import categorical_crossentropy

from DataGenerator import DataGenerator
from ModelUtils import create_model_class, parse_epoch
from model_efficient import Model_Efficient
from model_convnext import Model_ConvNeXt

# Directory constants
SNAPSHOTS_PATH = "snapshots"
OUTPUTS_PATH = "outputs"
MODELS_PATH = "models"  # New constant for models directory
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
BATCH_SIZE = 16
SEGMENTATION = False

np.random.seed(1337)  # for reproducibility


def finetune(args):
    # Create necessary directories
    for directory in [OUTPUTS_PATH, MODELS_PATH]:
        if not os.path.exists(directory):
            os.mkdir(directory)

    # Create timestamped model directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model
    model_dir = os.path.join(MODELS_PATH, f"{model_name}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)

    train_data_generator = DataGenerator(args.data_root, "train", 90, (0.7, 1.2), (1, 3), 0.3, 0.3, 5, BATCH_SIZE,
                                         (IMAGE_HEIGHT, IMAGE_WIDTH), args.debug_epochs, True, True, True, SEGMENTATION)
    val_data_generator = DataGenerator(args.data_root, "val", 90, (0.7, 1.2), (1, 3), 0.3, 0.3, 5, BATCH_SIZE,
                                         (IMAGE_HEIGHT, IMAGE_WIDTH), 0, True, True, True, SEGMENTATION)

    num_classes = train_data_generator.get_num_classes()

    # creating model
    model_name = args.model
    model_obj = create_model_class(model_name)
    model = model_obj.create_model(IMAGE_WIDTH, IMAGE_HEIGHT, num_classes)

    # preparing directories for snapshots
    if not os.path.exists(SNAPSHOTS_PATH):
        os.mkdir(SNAPSHOTS_PATH)

    model_snapshot_path = os.path.join(SNAPSHOTS_PATH, model_name)
    if not os.path.exists(model_snapshot_path):
        os.mkdir(model_snapshot_path)

    # saving labels to ints mapping
    train_data_generator.dump_labels_to_int_mapping(os.path.join(model_snapshot_path, "labels.csv"))

    start_epoch = 0
    if args.snapshot is not None:
        start_epoch = parse_epoch(args.snapshot)
        print("loading weights from epoch %d" % start_epoch)
        model.load_weights(os.path.join(model_snapshot_path, args.snapshot), by_name=True)

    # print summary
    model.summary()

    nb_epoch = 800
    sgd = optimizers.Adam(learning_rate=1e-5,  beta_1=0.9)#decay=1e-4,

    # selecting loss functions and weights
    losses = {}
    loss_weights = {}
    metrics = {}

    if SEGMENTATION:
        # segmentation and classification mode
        losses['segm_out'] = categorical_crossentropy
        loss_weights['segm_out'] = 1.0
        metrics['segm_out'] = 'accuracy'
        losses['class_out'] = categorical_crossentropy
        loss_weights['class_out'] = 1.0
        metrics['class_out'] = 'accuracy'
    else:
        # plain classification mode
        losses = 'categorical_crossentropy'
        loss_weights = None
        metrics = ['accuracy']

    model.compile(loss=losses, optimizer=sgd, metrics=metrics, loss_weights=loss_weights)
    
    # Save model configuration
    config_path = os.path.join(model_dir, "model_config.txt")
    with open(config_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Image size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Learning rate: 1e-5\n")
        f.write(f"Training started: {timestamp}\n")

    # Update paths to use new directory structure
    checkpoint_path = os.path.join(model_dir, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)
    
    filepath = os.path.join(checkpoint_path, "weights-{epoch:03d}-{accuracy:.3f}.keras")
    checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
        
    logpath = os.path.join(model_dir, f"{model_name}_training_log.txt")
    csv_logger = CSVLogger(logpath)

    # log dir for tensorboard
    tb_log_dir = os.path.join(model_dir, "tensorboard")
    os.makedirs(tb_log_dir, exist_ok=True)
    tb_log = TensorBoard(tb_log_dir)

    callbacks_list = [checkpoint, csv_logger, tb_log]

    # Train model
    history = model.fit(
        x=train_data_generator.generate(),
        steps_per_epoch=train_data_generator.get_steps_per_epoch(),
        epochs=nb_epoch,
        callbacks=callbacks_list,
        validation_data=val_data_generator.generate(),
        validation_steps=val_data_generator.get_steps_per_epoch(),
        initial_epoch=start_epoch
    )

    # Save final model and weights with descriptive names
    final_model_name = f"{model_name}_final_{timestamp}"
    final_model_path = os.path.join(model_dir, f"{final_model_name}.keras")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    final_weights_path = os.path.join(model_dir, f"{final_model_name}_weights.keras")
    model.save_weights(final_weights_path)
    print(f"Final weights saved to: {final_weights_path}")

    # Save training history
    history_path = os.path.join(model_dir, f"{model_name}_history.txt")
    with open(history_path, "w") as f:
        f.write(str(history.history))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Finetune model')
    parser.add_argument("data_root", type=str, help="data root dir")
    parser.add_argument("--model", type=str, help="name of the model")
    parser.add_argument("--snapshot", type=str, help="restart from snapshot")
    parser.add_argument("--debug_epochs", type=int, default=0, help="number of epochs to save debug images")

    _args = parser.parse_args()
    finetune(_args)