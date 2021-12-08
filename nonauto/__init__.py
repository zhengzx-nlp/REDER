import copy
import io
import os
import sys

from tensorboardX.record_writer import register_writer_factory

import nonauto.criterions
import nonauto.models
import nonauto.modules
import nonauto.tasks




class HDFSRecordWriter(object):
    """Writes tensorboard protocol buffer files to HDFS."""

    def __init__(self, path):

        self.path = path
        self.buffer = io.BytesIO()

    def __del__(self):
        self.close()

    def bucket_and_path(self):
        path = self.path
        if path.startswith("hdfs://"):
            path = path[len("hdfs://"):]
        bp = path.split("/")
        bucket = bp[0]
        path = path[1 + len(bucket):]
        return bucket, path

    def write(self, val):
        self.buffer.write(val)

    def flush(self):
        try:
            import tensorflow as tf
        except Exception as e:
            print("please install tensorflow")
            raise e
        upload_buffer = copy.copy(self.buffer)
        upload_buffer.seek(0)
        with tf.io.gfile.GFile(self.path, 'wb') as f:
            f.write(upload_buffer.getvalue())

    def close(self):
        self.flush()

class HDFSRecordWriterFactory(object):
    """Factory for event protocol buffer files to HDFS."""

    def open(self, path):
        return HDFSRecordWriter(path)

    def directory_check(self, path):
        # HDFS doesn't need directories created before files are added
        # so we can just skip this check
        pass

register_writer_factory("hdfs", HDFSRecordWriterFactory())

