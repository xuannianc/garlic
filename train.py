from keras.optimizers import SGD
from model.vgg import East
from data.hdf5 import HDF5DatasetGenerator
from callbacks import *
from keras import callbacks
from losses import loss
from keras.models import load_model
from matplotlib import pyplot as plt

opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
east = East()
# model = east.build()
# model.compile(loss=loss, optimizer=opt)
model = load_model('saved_models/east_1027_7900_1976_6.5662_7.1822.hdf5', custom_objects={'loss': loss})
print('old__lr={}'.format(K.get_value(model.optimizer.lr)))
K.set_value(model.optimizer.lr, 0.001)
print('new__lr={}'.format(K.get_value(model.optimizer.lr)))
# callbacks
training_monitor = TrainingMonitor(figure_path='output/east_7900_1976.jpg',
                                   json_path='output/east_7900_1976.json',
                                   start_at=7)
# accuracy_evaluator = AccuracyEvaluator(TEST_DB_PATH, batch_size=100)
# learning_rate_updator = LearningRateUpdator(init_lr=0.001)
callbacks = [
    # Interrupts training when improvement stops
    callbacks.EarlyStopping(
        # Monitors the model’s validation accuracy
        monitor='val_loss',
        # Interrupts training when accuracy has stopped
        # improving for more than one epoch (that is, two epochs)
        patience=5,
    ),
    # Saves the current weights after every epoch
    callbacks.ModelCheckpoint(
        # Path to the destination model file
        filepath='saved_models/east_1027_7900_1976.hdf5',
        # These two arguments mean you won’t overwrite the
        # model file unless val_loss has improved, which allows
        # you to keep the best model seen during training.
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    training_monitor,
    # accuracy_evaluator
    # learning_rate_updator
]
train_gen = HDF5DatasetGenerator('data/train_7900.hdf5', batch_size=1).generator
val_gen = HDF5DatasetGenerator('data/val_1976.hdf5', batch_size=1).generator
H = model.fit_generator(train_gen(), steps_per_epoch=7900,
                        callbacks=callbacks,
                        epochs=100,
                        validation_data=val_gen(),
                        validation_steps=1976)
N = np.arange(0, len(H.history["loss"]))
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
