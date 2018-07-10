import * as tf from '@tensorflow/tfjs';
import {
  IParams,
  IParamsCallbacks,
} from './types';

interface ITrainingOpts {
  xs?: tf.Tensor3D;
  ys?: tf.Tensor2D;
  classes: number;
}

const defaultLayers = ({ classes }: ITrainingOpts) => [
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

const getBatchSize = (params: IParams, xs?: tf.Tensor3D) => {
  if (params.batchSize) {
    return params.batchSize;
  }

  if (xs !== undefined) {
    return Math.floor(xs.shape[0] * 0.4);
  }

  return undefined;
};

const transformCallbacks = (callbacks: IParamsCallbacks = {}) => Object.entries(callbacks).reduce((callbackObj, [
  key,
  callback,
]) => {
  if (callback) {
    return {
      ...callbackObj,
      [key]: async (...args: any[]) => {
        callback(...args);
        await tf.nextFrame();
      }
    };
  }

  return callbackObj;
}, {});

const train = async ({
  xs,
  ys,
  classes,
}: ITrainingOpts, params: IParams) => {
  if (xs === undefined || ys === undefined) {
    throw new Error('Add some examples before training!');
  }

  const layers = params.layers || defaultLayers;

  const model = tf.sequential({
    layers: layers({ xs, ys, classes }),
  });

  model.compile({
    optimizer: params.optimizer,
    loss: params.loss,
    metrics: ['accuracy'],
  });

  const batchSize = getBatchSize(params, xs);

  await model.fit(
    xs,
    ys,
    {
      batchSize,
      epochs: params.epochs,
      callbacks: transformCallbacks({
        onTrainBegin: params.callbacks.onTrainBegin,
        onTrainEnd: params.callbacks.onTrainEnd,
        onEpochBegin: params.callbacks.onEpochBegin,
        onEpochEnd: params.callbacks.onEpochEnd,
        onBatchBegin: params.callbacks.onBatchBegin,
        onBatchEnd: params.callbacks.onBatchEnd,
      }),
      validationSplit: params.validationSplit,
      validationData: params.validationData,
      shuffle: params.shuffle,
      classWeight: params.classWeight,
      sampleWeight: params.sampleWeight,
      initialEpoch: params.initialEpoch,
      stepsPerEpoch: params.stepsPerEpoch,
      validationSteps: params.validationSteps,
      verbose: params.verbose,
    },
  );

  return model;
};

export default train;
