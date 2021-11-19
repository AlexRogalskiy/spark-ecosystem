import shutil
import uuid

import numpy as np

import pyspark.sql.types as T
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder
from pyspark.sql.functions import udf

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import horovod.spark.torch as hvd
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
# Define the PyTorch model without any Horovod-specific parameters


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.float()
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
loss = nn.NLLLoss()
# Train a Horovod Spark Estimator on the DataFrame
torch_estimator = hvd.TorchEstimator(
    num_proc=num_proc,
    store=store,
    model=model,
    optimizer=optimizer,
    loss=lambda input, target: loss(input, target.long()),
    input_shapes=[[-1, 1, 28, 28]],
    feature_cols=['features'],
    label_cols=['label'],
    batch_size=batch_size,
    epochs=epochs,
    verbose=1)

torch_model = torch_estimator.fit(train_df).setOutputCols(['label_prob'])
# Evaluate the model on the held-out test DataFrame
pred_df = torch_model.transform(test_df)
argmax = udf(lambda v: float(np.argmax(v)), returnType=T.DoubleType())
pred_df = pred_df.withColumn('label_pred', argmax(pred_df.label_prob))
evaluator = MulticlassClassificationEvaluator(
    predictionCol='label_pred', labelCol='label', metricName='accuracy')
print('Test accuracy:', evaluator.evaluate(pred_df))
shutil.rmtree(work_dir)
