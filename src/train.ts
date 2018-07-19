import * as tf from '@tensorflow/tfjs';
import {
  IParams,
  IImageData,
  IArgs,
} from './types';

const defaultLayers = ({ classes }: { classes: number }) => {
  return [
    tf.layers.flatten({inputShape: [7, 7, 256]}),
    tf.layers.dense({
      units: 100,
      activation: 'relu',
      kernelInitializer: 'varianceScaling',
      useBias: true
    }),
    tf.layers.dense({
      units: classes,
      kernelInitializer: 'varianceScaling',
      useBias: false,
      activation: 'softmax'
    })
  ];
};

const getBatchSize = (batchSize?: number, xs?: tf.Tensor) => {
  if (batchSize) {
    return batchSize;
  }

  if (xs !== undefined) {
    return Math.floor(xs.shape[0] * 0.4) || 1;
  }

  return undefined;
};

const getModel = (pretrainedModel: tf.Model, data: IImageData, classes: number, params: IParams, args: IArgs) => {
  if (args.trainingModel) {
    if (typeof args.trainingModel === 'function') {
      return args.trainingModel(data, classes, params);
    }

    return args.trainingModel;
  }

  const model = tf.sequential({
    layers: defaultLayers({ classes }),
  });

  const optimizer = tf.train.adam(0.0001);

  model.compile({
    optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
};

const train = async (pretrainedModel: tf.Model, data: IImageData, classes: number, params: IParams, args: IArgs) => {
  const {
    xs,
    ys,
  } = data;

  if (xs === undefined || ys === undefined) {
    throw new Error('Add some examples before training!');
  }

  // const batch = data.nextTrainBatch(BATCH_SIZE);

  const model = getModel(pretrainedModel, data, classes, params, args);

  const batchSize = getBatchSize(params.batchSize, xs);

  const history = await model.fit(
    xs,
    ys,
    {
      ...params,
      batchSize,
      epochs: params.epochs || 20,
    },
  );

  return {
    model,
    history,
  };
};

export default train;
