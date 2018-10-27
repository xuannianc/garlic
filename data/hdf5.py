import h5py
import os
import numpy as np
import logging
import sys

logger = logging.getLogger('hdf5')
logger.setLevel(logging.DEBUG)  # default log level
format = logging.Formatter("%(asctime)s %(name)-8s %(levelname)-8s %(lineno)-4d %(message)s")  # output format
sh = logging.StreamHandler(stream=sys.stdout)  # output to standard output
sh.setFormatter(format)
logger.addHandler(sh)


class HDF5DatasetWriter:
    def __init__(self, image_dims, label_dims, output_path, buffer_size=10):
        # check to see if the output path exists, and if so, raise an exception
        if os.path.exists(output_path):
            raise ValueError(
                "{} already exists and cannot be overwritten. Manually delete the file before continuing.".format(
                    output_path))
        # open the HDF5 database for writing and create two datasets:
        # one to store the images/features and another to store the
        # class labels
        self.db = h5py.File(output_path, "w")
        self.images = self.db.create_dataset('images', image_dims, dtype="float")
        self.labels = self.db.create_dataset("labels", label_dims, dtype="int")
        # store the buffer size, then initialize the buffer itself
        # along with the index into the datasets
        self.buffer_size = buffer_size
        self.buffer = {"images": [], "labels": []}
        # idx of hdf5 databases
        self.idx = 0

    def add(self, rows, labels):
        # add the rows and labels to the buffer
        self.buffer["images"].extend(rows)
        self.buffer["labels"].extend(labels)
        # check to see if the buffer needs to be flushed to disk
        if len(self.buffer["images"]) >= self.buffer_size:
            self.flush()

    def flush(self):
        # write the buffers to disk then reset the buffer
        logger.debug('Flushing starts...')
        i = self.idx + len(self.buffer["images"])
        self.images[self.idx:i] = self.buffer["images"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"images": [], "labels": []}
        logger.debug('Flushing ends...')

    def close(self):
        # check to see if there are any other entries in the buffer
        # that need to be flushed to disk
        if len(self.buffer["images"]) > 0:
            self.flush()
        # close the dataset
        self.db.close()


class HDF5DatasetGenerator:
    def __init__(self, db_path, batch_size):
        # store the batch size, preprocessors, and data augmentor,
        # whether or not the labels should be binarized, along with
        # the total number of classes
        self.batch_size = batch_size
        # open the HDF5 database for reading and determine the total
        # number of entries in the database
        self.db = h5py.File(db_path)
        self.num_images = self.db["images"].shape[0]

    def generator(self):
        # keep looping infinitely
        while True:
            # loop over the HDF5 dataset
            for i in np.arange(0, self.num_images, self.batch_size):
                # extract the images and labels from the HDF dataset
                images = self.db["images"][i:i + self.batch_size]
                labels = self.db["labels"][i:i + self.batch_size]
                yield images, labels

    def close(self):
        # close the database
        self.db.close()
