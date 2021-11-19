import os
import shutil
import uuid
from distutils.version import LooseVersion

import numpy as np

import pyspark.sql.types as T
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder
from pyspark.sql.functions import udf

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import horovod.spark.keras as hvd
from horovod.spark.common.store import DBFSLocalStore
uuid_str = str(uuid.uuid4())
work_dir = "/dbfs/horovod_spark_estimator/"+uuid_str
num_proc = 2  # num_proc < (# worker CPUs) or (# worker GPUs)
batch_size = 5
epochs = 2
# Setup store for intermediate data
store = DBFSLocalStore(work_dir)

# Load MNIST data from databricks-datasets
# So that this notebook can run quickly, this example uses the .limit() option. Using only limited data decreases the model's accuracy; remove this option for better accuracy.
train_df = spark.read.format("libsvm") \
    .option('numFeatures', '784') \
    .load("/databricks-datasets/mnist-digits/data-001/mnist-digits-train.txt") \
    .limit(60).repartition(num_proc)
test_df = spark.read.format("libsvm") \
    .option('numFeatures', '784') \
    .load("/databricks-datasets/mnist-digits/data-001/mnist-digits-test.txt") \
    .limit(20).repartition(num_proc)
# One-hot encode labels into SparseVectors
encoder = OneHotEncoder(inputCols=['label'],
                        outputCols=['label_vec'],
                        dropLast=False)
model = encoder.fit(train_df)
train_df = model.transform(train_df)
test_df = model.transform(test_df)
# Disable GPUs when building the model to prevent memory leaks
if LooseVersion(tf.__version__) >= LooseVersion('2.0.0'):
    # See https://github.com/tensorflow/tensorflow/issues/33168
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
else:
    keras.backend.set_session(tf.Session(
        config=tf.ConfigProto(device_count={'GPU': 0})))
# Define the Keras model without any Horovod-specific parameters
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

optimizer = keras.optimizers.Adadelta(1.0)
loss = keras.losses.categorical_crossentropy
# Train a Horovod Spark Estimator on the DataFrame
keras_estimator = hvd.KerasEstimator(
    num_proc=num_proc,
    store=store,
    model=model,
    optimizer=optimizer,
    loss=loss,
    metrics=['accuracy'],
    feature_cols=['features'],
    label_cols=['label_vec'],
    batch_size=batch_size,
    epochs=epochs,
    verbose=1)

keras_model = keras_estimator.fit(train_df).setOutputCols(['label_prob'])
num_partitions = 40
train_data_path = file: // /dbfs/horovod_spark_estimator/fb335f73-8ed0-4580-ae55-2fa4f9568255/intermediate_train_data.0
val_data_path = file: // /dbfs/horovod_spark_estimator/fb335f73-8ed0-4580-ae55-2fa4f9568255/intermediate_val_data.0
train_partitions = 40
metadata, avg_row_size = make_metadata_dictionary(train_data_schema)

# Evaluate the model on the held-out test DataFrame
pred_df = keras_model.transform(test_df)
argmax = udf(lambda v: float(np.argmax(v)), returnType=T.DoubleType())
pred_df = pred_df.withColumn('label_pred', argmax(pred_df.label_prob))
evaluator = MulticlassClassificationEvaluator(
    predictionCol='label_pred', labelCol='label', metricName='accuracy')
print('Test accuracy:', evaluator.evaluate(pred_df))
shutil.rmtree(work_dir)
