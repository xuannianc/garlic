# import the necessary packages
from keras.callbacks import BaseLogger, Callback
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from keras import backend as K
from keras.models import Model


class TrainingMonitor(BaseLogger):
    def __init__(self, figure_path, json_path=None, start_at=0):
        # store the output path for the figure, the path to the JSON
        # serialized file, and the starting epoch
        super(TrainingMonitor, self).__init__()
        self.figure_path = figure_path
        self.json_path = json_path
        self.start_at = start_at

    def on_train_begin(self, logs={}):
        # initialize the history dictionary
        self.H = {}
        # if the JSON history path exists, load the training history
        if self.json_path is not None:
            if os.path.exists(self.json_path):
                self.H = json.loads(open(self.json_path).read())
                # check to see if a starting epoch was supplied
                if self.start_at > 0:
                    # loop over the entries in the history log and
                    # trim any entries that are past the starting
                    # epoch
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.start_at]
                else:
                    self.H = {}

    def on_epoch_end(self, epoch, logs={}):
        # loop over the logs and update the loss, accuracy, etc.
        # for the entire training process
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l
        # check to see if the training history should be serialized
        # to file
        if self.json_path is not None:
            f = open(self.json_path, "w")
            f.write(json.dumps(self.H))
            f.close()

        # ensure at least two epochs have passed before plotting
        # (epoch starts at zero)
        if len(self.H["loss"]) > 1:
            # plot the training loss and accuracy
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["loss"], label="train_loss")
            plt.plot(N, self.H["val_loss"], label="val_loss")
            plt.title("Training Loss [Epoch {}]".format(
                len(self.H["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            # save the figure
            plt.savefig(self.figure_path)
            plt.close()


class LearningRateUpdator(Callback):
    def __init__(self, init_lr):
        super(LearningRateUpdator, self).__init__()
        self.init_lr = init_lr

    def on_train_begin(self, logs=None):
        old_lr = K.get_value(self.model.optimizer.lr)
        print('old_init_lr={}'.format(old_lr))
        K.set_value(self.model.optimizer.lr, self.init_lr)
        print('new_init_lr={}'.format(K.get_value(self.model.optimizer.lr)))

    def on_epoch_end(self, epoch, logs=None):
        model = self.model
        old_lr = K.get_value(model.optimizer.lr)
        print('old_lr={},epoch={}'.format(old_lr, epoch))
        new_lr = old_lr - self.init_lr * 0.01
        K.set_value(model.optimizer.lr, new_lr)
        print('new_lr={},epoch={}'.format(K.get_value(model.optimizer.lr), epoch))
