import * as tf from '@tensorflow/tfjs';
import {
  IParams,
  IImageData,
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

const getBatchSize = (batchSize?: number, xs?: tf.Tensor3D) => {
  if (batchSize) {
    return batchSize;
  }

  if (xs !== undefined) {
    return Math.floor(xs.shape[0] * 0.4);
  }

  return undefined;
};

const train = async ({
  xs,
  ys,
}: IImageData, classes: number, params: IParams) => {
  if (xs === undefined || ys === undefined) {
    throw new Error('Add some examples before training!');
  }

  // const batch = data.nextTrainBatch(BATCH_SIZE);

  const model = tf.sequential({
    layers: defaultLayers({ classes }),
  });

  const optimizer = tf.train.adam(0.0001);

  model.compile({
    optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

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
