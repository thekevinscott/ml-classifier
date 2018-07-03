import * as tf from '@tensorflow/tfjs';

interface IOpts {
  onTrainBegin?: Function;
  onTrainEnd?: Function;
  onEpochBegin?: Function;
  onEpochEnd?: Function;
  onBatchBegin?: Function;
  onBatchEnd?: Function;
}

const train = async ({
  xs,
  ys,
  classes,
}: {
  xs: any;
  ys: any;
  classes: number;
}, callbacks: IOpts) => {
  if (xs === null) {
    throw new Error('Add some examples before training!');
  }

  // Creates a 2-layer fully connected model. By creating a separate model,
  // rather than adding layers to the mobilenet model, we "freeze" the weights
  // of the mobilenet model, and only train weights from the new model.
  const model = tf.sequential({
    layers: [
      // Flattens the input to a vector so we can use it in a dense layer. While
      // technically a layer, this only performs a reshape (and has no training
      // parameters).
      tf.layers.flatten({inputShape: [7, 7, 256]}),
      // Layer 1
      tf.layers.dense({
        units: 100,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true
      }),
      // Layer 2. The number of units of the last layer should correspond
      // to the number of classes we want to predict.
      tf.layers.dense({
        units: classes,
        kernelInitializer: 'varianceScaling',
        useBias: false,
        activation: 'softmax'
      })
    ]
  });

  // Creates the optimizers which drives training of the model.
  const optimizer = tf.train.adam(0.0001);
  // We use categoricalCrossentropy which is the loss function we use for
  // categorical classification which measures the error between our predicted
  // probability distribution over classes (probability that an input is of each
  // class), versus the label (100% probability in the true class)>
  model.compile({ optimizer, loss: 'categoricalCrossentropy' });

  // We parameterize batch size as a fraction of the entire dataset because the
  // number of examples that are collected depends on how many examples the user
  // collects. This allows us to have a flexible batch size.
  const batchSize = Math.floor(xs.shape[0] * 0.4);

  await model.fit(
    xs,
    ys,
    {
      batchSize,
      epochs: 20,
      callbacks: Object.entries(callbacks).reduce((callbackObj, [key, callback]) => {
        return {
          ...callbackObj,
          [key]: async (...args: any[]) => {
            callback(...args);
            await tf.nextFrame();
          }
        };
      }, {}),
    },
  );

  return model;
};

export default train;
