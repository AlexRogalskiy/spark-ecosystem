### BigDL

https://github.com/intel-analytics/BigDL

#### 什么是 BigDL？

BigDL 是一款面向 Spark 的分布式深度学习库，在现有的 Spark 或 Apache Hadoop* 集群上直接运行。您可以将深度学习应用编写为 Scala 或 Python 程序。

- **丰富的深度学习支持。[BigDL](https://github.com/intel-analytics/BigDL)** 模仿 [Torch](http://torch.ch/) ，为深度学习提供综合支持，包括数值计算（借助[Tensor](https://github.com/intel-analytics/BigDL/tree/master/dl/src/main/scala/com/intel/analytics/bigdl/tensor)）和[高级神经网络](https://github.com/intel-analytics/BigDL/tree/master/dl/src/main/scala/com/intel/analytics/bigdl/nn)；此外，用户可以将预训练 [Caffe](http://caffe.berkeleyvision.org/)* 或 Torch 模型加载至 Spark 框架，并使用 BigDL 库在数据中运行推断应用。
- **高效的横向扩展。**利用 [Spark](http://spark.apache.org/)，BigDL 能够在 Spark 中高效地横向扩展，处理大数据规模的数据分析，高效实施随机梯度下降 (SGD)，以及进行 all-reduce 通信。
- **极高的性能。**为了实现较高的性能，BigDL 在每个 Spark 任务中采用[英特尔® 数学核心函数库](https://software.intel.com/zh-cn/intel-mkl)（英特尔® MKL）和多线程编程。因此，相比现成的开源 Caffe、Torch 或 [TensorFlow](https://www.tensorflow.org/)，BigDL 在单节点英特尔® 至强® 处理器上的运行速度高出多个数量级（与主流图形处理单元相当）。

#### 什么是 Apache Spark*？

Spark 是一款极速的分布式数据处理框架，由加利福尼亚大学伯克利分校的 AMPLab 开发。Spark 可以以独立模式运行，也能以集群模式在 Hadoop 上的 YARN 中或 Apache Mesos* 集群管理器上运行（图 2）。Spark 可以处理各种来源的数据，包括 HDFS、Apache Cassandra* 或 Apache Hive*。由于它能够通过持久存储的 RDD 或 DataFrames 处理内存，而不是将数据保存至硬盘（如同传统的 Hadoop MapReduce 架构），因此，极大地提高了性能。

![img](https://software.intel.com/content/dam/develop/external/us/en/images/bigdl-on-apache-spark-fig-02-stack-712489.png)
**图 2.** Apache Spark* 堆栈中的 BigDL

#### 为什么使用 BigDL？

在以下情况下，您需要利用 BigDL 编写您的深度学习程序：

- 您希望在存储数据的大数据 Spark 集群（如 HDFS、Apache HBase* 或 Hive ）上分析大量数据；
- 您希望在大数据 (Spark) 程序或工作流中添加深度学习功能（训练或预测）；或者
- 您希望利用现有的 Hadoop/Spark 集群运行深度学习应用，随后与其他工作负载轻松共享（例如提取-转换-加载、数据仓库、特性设计、经典机器学习、图形分析）。另一种使用 BigDL 的不常见的替代方案是与 Spark 同时引进另一种分布式框架，以实施深度学习算法。