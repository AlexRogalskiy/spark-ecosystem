### deeplearning4j

　　deeplearning4j是由Skymind开发的，Skymind是一家致力于为企业进行商业化深度学习的公司。deeplearning4j框架是创建来在Hadoop及Spark上运行的。这个设计用于商业环境而不是许多深度学习框架及库目前所大量应用的研究领域。Skymind是主要的支持者，但deeplearning4j是开源软件，因此也欢迎大家提交补丁。deeplearning4j框架中实现了如下算法：

- 受限玻尔兹曼机（Restricted Boltzmann Machine）
- 卷积神经网络（Convolutional Neural Network）
- 循环神经网络（Recurrent Neural Network）
- 递归自编码器（Recursive Autoencoder）
- 深度信念网络（Deep-Belief Network）
- 深度自编码器（Deep Autoencoder）
- 栈式降噪自编码（Stacked Denoising Autoencoder）

这里要注意的是，这些模型能在细粒度级别进行配置。你可以设置隐藏的层数、每个神经元的激活函数以及迭代的次数。deeplearning4j提供了不同种类的网络实现及灵活的模型参数。Skymind也开发了许多工具，对于更稳定地运行机器学习算法很有帮助。下面列出了其中的一些工

**Canova [https://github.com/deeplearning4j/Canoba]**是一个向量库。机器学习算法能以向量格式处理所有数据。所有的图片、音频及文本数据必须用某种方法转换为向量。虽然训练机器学习模型是十分常见的工作，但它会重新造轮子还会引起bug。Canova能为你做这种转换。Canova当前支持的输入数据格式为：
-- CSV
--原始文本格式（推文、文档）
--图像（图片、图画）
--定制文件格式（例如MNIST）

- **由于Canova主要是用Java编写的，所以它能运行在所有的JVM平台上。**因此，可以在Spark集群上使用它。即使你不做机器学习，Canova对你的机器学习任务可能也会有所裨益。
- **nd4j** [https://github.com/deeplearning4j/nd4j] **有点像是一个numpy，Python中的SciPy工具。**此工具提供了线性代数、向量计算及操纵之类的科学计算。它也是用Java编写的。你可以根据自己的使用场景来搭配使用这些工具。需要注意的一点是，nd4j支持GPU功能。由于现代计算硬件还在不断发展，有望达到更快速的计算。
- **dl4j-spark-ml** [https://github.com/deeplearning4j/dl4j-spark-ml]** 是一个Spark包，使你能在Spark上轻松运行deeplearning4j。**使用这个包，就能轻松在Spark上集成deeplearning4j，因为它已经被上传到了Spark包的公共代码库。

因此，如果你要在Spark上使用deeplearning4j，我们推荐通过dl4j-spark-ml包来实现。与往常一样，必须下载或自己编译Spark源码。这里对Spark版本没有特别要求，就算使用最早的版本也可以。deeplearning4j项目准备了样例存储库。要在Spark上使用deeplearning4j，dl4j-Spark-ml-examples是可参考的最佳示例（https:// github.com/deeplearning4j/dl4j-Spark-ml-examples）