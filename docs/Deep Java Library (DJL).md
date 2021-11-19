

# Deep Java Library (DJL)

https://github.com/deepjavalibrary/djl

## Overview

Deep Java Library (DJL) is an open-source, high-level, engine-agnostic Java framework for deep learning. DJL is designed to be easy to get started with and simple to use for Java developers. DJL provides a native Java development experience and functions like any other regular Java library.

You don't have to be machine learning/deep learning expert to get started. You can use your existing Java expertise as an on-ramp to learn and use machine learning and deep learning. You can use your favorite IDE to build, train, and deploy your models. DJL makes it easy to integrate these models with your Java applications.

Because DJL is deep learning engine agnostic, you don't have to make a choice between engines when creating your projects. You can switch engines at any point. To ensure the best performance, DJL also provides automatic CPU/GPU choice based on hardware configuration.

DJL's ergonomic API interface is designed to guide you with best practices to accomplish deep learning tasks. The following pseudocode demonstrates running inference:

```java
    // Assume user uses a pre-trained model from model zoo, they just need to load it
    Criteria<Image, Classifications> criteria =
            Criteria.builder()
                    .optApplication(Application.CV.OBJECT_DETECTION) // find object dection model
                    .setTypes(Image.class, Classifications.class) // define input and output
                    .optFilter("backbone", "resnet50") // choose network architecture
                    .build();

    try (ZooModel<Image, Classifications> model = criteria.loadModel()) {
        try (Predictor<Image, Classifications> predictor = model.newPredictor()) {
            Image img = ImageFactory.getInstance().fromUrl("http://..."); // read image
            Classifications result = predictor.predict(img);

            // get the classification and probability
            ...
        }
    }
```

The following pseudocode demonstrates running training:

```java
    // Construct your neural network with built-in blocks
    Block block = new Mlp(28, 28);

    try (Model model = Model.newInstance("mlp")) { // Create an empty model
        model.setBlock(block); // set neural network to model

        // Get training and validation dataset (MNIST dataset)
        Dataset trainingSet = new Mnist.Builder().setUsage(Usage.TRAIN) ... .build();
        Dataset validateSet = new Mnist.Builder().setUsage(Usage.TEST) ... .build();

        // Setup training configurations, such as Initializer, Optimizer, Loss ...
        TrainingConfig config = setupTrainingConfig();
        try (Trainer trainer = model.newTrainer(config)) {
            /*
             * Configure input shape based on dataset to initialize the trainer.
             * 1st axis is batch axis, we can use 1 for initialization.
             * MNIST is 28x28 grayscale image and pre processed into 28 * 28 NDArray.
             */
            Shape inputShape = new Shape(1, 28 * 28);
            trainer.initialize(new Shape[] {inputShape});

            EasyTrain.fit(trainer, epoch, trainingSet, validateSet);
        }

        // Save the model
        model.save(modelDir, "mlp");
    }
```