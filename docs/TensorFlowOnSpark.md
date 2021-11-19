### TensorFlowOnSpark

https://github.com/yahoo/TensorFlowOnSpark

TensorFlowOnSpark 为 Apache Hadoop 和 Apache Spark 集群带来可扩展的深度学习。 通过结合深入学习框架 TensorFlow 和大数据框架 Apache Spark 、Apache Hadoop 的显着特征，TensorFlowOnSpark 能够在 GPU 和 CPU 服务器集群上实现分布式深度学习。

**TensorFlowOnSpark**

![img](https://static001.infoq.cn/resource/image/9b/dc/9bae4abc1c69491d645975b3f88137dc.png)

我们的新框架 TensorFlowOnSpark（TFoS），支持 TensorFlow 在 Spark 和 Hadoop 集群上分布式执行。如上图 2 所示，TensorFlowOnSpark 被设计为与 SparkSQL、MLlib 和其他 Spark 库一起在一个单独流水线或程序（如 Python notebook）中运行。

TensorFlowOnSpark 支持所有类型的 TensorFlow 程序，可以实现异步和同步的训练和推理。它支持模型并行性和数据的并行处理，以及 TensorFlow 工具（如 Spark 集群上的 TensorBoard）。

任何 TensorFlow 程序都可以轻松地修改为在 TensorFlowOnSpark 上运行。通常情况下，需要改变的 Python 代码少于 10 行。许多 Yahoo 平台使用 TensorFlow 的开发人员很容易迁移 TensorFlow 程序，以便在 TensorFlowOnSpark 上执行。

TensorFlowOnSpark 支持 TensorFlow 进程（计算节点和参数服务节点）之间的直接张量通信。过程到过程的直接通信机制使 TensorFlowOnSpark 程序能够在增加的机器上很轻松的进行扩展。如图 3 所示，TensorFlowOnSpark 不涉及张量通信中的 Spark 驱动程序，因此实现了与独立 TensorFlow 集群类似的可扩展性。

![img](https://static001.infoq.cn/resource/image/c7/77/c7406e478beb085693c0e431f5f53c77.png)

TensorFlowOnSpark 提供两种不同的模式来提取训练和推理数据：

1. **TensorFlow QueueRunners：**TensorFlowOnSpark 利用 TensorFlow 的[ file readers ](http://t.umblr.com/redirect?z=https://www.tensorflow.org/how_tos/reading_data/#reading_from_files&t=MDk1NzhlZDQ0YTM2MTY4OWY4OTFhOGYzMDRjYmMxOGY4N2NiMmY3Myx1VmR1UG1vZg==&b=t%3afgAkOE96nMUZDZ4JRZ0Fgw&p=http://yahoohadoop.tumblr.com/post/157196317141/open-sourcing-tensorflowonspark-distributed-deep&m=1)和[ QueueRunners ](http://t.umblr.com/redirect?z=https://www.tensorflow.org/how_tos/threading_and_queues/#queuerunner&t=ZjI4YjM5ODg4NTZiMmVlMTNjN2JhOWEyNzdkMjk5NjE0MTFiOTdlMix1VmR1UG1vZg==&b=t%3afgAkOE96nMUZDZ4JRZ0Fgw&p=http://yahoohadoop.tumblr.com/post/157196317141/open-sourcing-tensorflowonspark-distributed-deep&m=1)直接从 HDFS 文件中读取数据。Spark 不涉及访问数据。
2. **Spark Feeding** ：Spark RDD 数据被传输到每个 Spark 执行器里，随后的数据将通过[ feed_dict ](http://t.umblr.com/redirect?z=https://www.tensorflow.org/how_tos/reading_data/#feeding&t=YWY2Y2U4YTE0ODc2M2E0NzYwNjFjZTE2MWE1ZWY5M2JjOTNiMTdlZCx1VmR1UG1vZg==&b=t%3afgAkOE96nMUZDZ4JRZ0Fgw&p=http://yahoohadoop.tumblr.com/post/157196317141/open-sourcing-tensorflowonspark-distributed-deep&m=1)传入 TensorFlow 图。

**简单的CLI 和API**

TFoS 程序由标准的 Apache Spark 命令 _spark-submit_ 来启动。如下图所示，用户可以在 CLI 中指定 Spark 执行器的数目，每个执行器的 GPU 数量和参数服务器的数目。用户还可以指定是否要使用 TensorBoard（-tensorboard）和 / 或 RDMA（-rdma）。

复制代码

```shell
 
     spark-submit –master ${MASTER} \ 
     ${TFoS_HOME}/examples/slim/train_image_classifier.py \ 
     –model_name inception_v3 \
     –train_dir hdfs://default/slim_train \ 
     –dataset_dir hdfs://default/data/imagenet \
     –dataset_name imagenet \
     –dataset_split_name train \
     –cluster_size ${NUM_EXEC} \
     –num_gpus ${NUM_GPU} \
     –num_ps_tasks ${NUM_PS} \
     –sync_replicas \
     –replicas_to_aggregate ${NUM_WORKERS} \
     –tensorboard \
     –rdma  
```

TFoS 提供了一个高层次的 Python API（在我[们示例 Python notebook ](http://t.umblr.com/redirect?z=https://github.com/yahoo/TensorFlowOnSpark/blob/master/examples/mnist/TFOS_demo.ipynb&t=MWFkZDEwZTExNDY1NDQ0ZTkwODgxODgzMmM0MTgwZTk1MTU4NzAwNSx1VmR1UG1vZg==&b=t%3afgAkOE96nMUZDZ4JRZ0Fgw&p=http://yahoohadoop.tumblr.com/post/157196317141/open-sourcing-tensorflowonspark-distributed-deep&m=1)说明）：

- TFCluster.reserve() … construct a TensorFlow cluster from Spark executors
- TFCluster.start() … launch Tensorflow program on the executors
- TFCluster.train() or TFCluster.inference() … feed RDD data to TensorFlow processes
- TFCluster.shutdown() … shutdown Tensorflow execution on executors

**开放源码**

[TensorFlowOnSpark ](http://t.umblr.com/redirect?z=https://github.com/yahoo/TensorFlowOnSpark&t=NjRhYmYzODNiNzQ1ODUwZjIwOGRiZDQyZmMyYThkMzExMmM2ZWNjOCx1VmR1UG1vZg==&b=t%3afgAkOE96nMUZDZ4JRZ0Fgw&p=http://yahoohadoop.tumblr.com/post/157196317141/open-sourcing-tensorflowonspark-distributed-deep&m=1)、[ TensorFlow 的 RDMA 增强包](http://t.umblr.com/redirect?z=https://github.com/yahoo/tensorflow/tree/yahoo&t=NWE0M2NjODYwOGMzM2I1MTNhZjUyZDQwMGU1ZDRmNmE3NjIxNzQwNCx1VmR1UG1vZg==&b=t%3afgAkOE96nMUZDZ4JRZ0Fgw&p=http://yahoohadoop.tumblr.com/post/157196317141/open-sourcing-tensorflowonspark-distributed-deep&m=1)、多个[示例程序](http://t.umblr.com/redirect?z=https://github.com/yahoo/TensorFlowOnSpark/tree/master/examples&t=OGVjN2VhM2UxZWQ3NDNiMDg4NTM5ODA0ZWI4YjQ2ODYxM2UxYzIyZix1VmR1UG1vZg==&b=t%3afgAkOE96nMUZDZ4JRZ0Fgw&p=http://yahoohadoop.tumblr.com/post/157196317141/open-sourcing-tensorflowonspark-distributed-deep&m=1)（包括MNIST，cifar10，创建以来，VGG）来说明TensorFlow 方案TensorFlowOnSpark，并充分利用RDMA 的简单转换过程。亚马逊机器映像也[可](http://t.umblr.com/redirect?z=https://github.com/yahoo/TensorFlowOnSpark/wiki/GetStarted_EC2&t=MTcyNGUyYjdjMTZkNWYyYjAwNGE5NGY3M2Q0ZTI5ZTc3ZDllMGVhZCx1VmR1UG1vZg==&b=t%3afgAkOE96nMUZDZ4JRZ0Fgw&p=http://yahoohadoop.tumblr.com/post/157196317141/open-sourcing-tensorflowonspark-distributed-deep&m=1)对AWS EC2 应用TensorFlowOnSpark。