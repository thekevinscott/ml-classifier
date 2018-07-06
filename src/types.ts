import * as tf from '@tensorflow/tfjs';

enum ModelLoggingVerbosity {
  SILENT = 0,
  VERBOSE = 1
}

export interface IImage {
  label: string;
  data: tf.Tensor3D;
}

export interface IActivatedImage {
  activation: tf.Tensor3D;
  label: string;
};

export interface ITrainingData {
  classes: {
    [index: string]: number;
  };
  xs?: tf.Tensor3D;
  ys?: tf.Tensor2D;
}

export interface IParamsCallbacks {
  onTrainBegin?: Function;
  onTrainEnd?: Function;
  onEpochBegin?: Function;
  onEpochEnd?: Function;
  onBatchBegin?: Function;
  onBatchEnd?: Function;
};

export interface IParams {
  optimizer: string | tf.Optimizer;
  loss: string | string[];
  layers?: Function;
  // layers?: ({
  //   classes: number;
  //   xs?: tf.Tensor3D;
  //   ys?: tf.Tensor2D;
  // }) => tf.layers.Layer[];
  model?: tf.Model;
  batchSize?: number;
  epochs?: number;
  callbacks: IParamsCallbacks;
  validationSplit?: number;
  validationData?: [ tf.Tensor|tf.Tensor[], tf.Tensor|tf.Tensor[] ]|[tf.Tensor | tf.Tensor[], tf.Tensor|tf.Tensor[], tf.Tensor|tf.Tensor[]];
  shuffle?: boolean;
  classWeight?: {[classIndex: string]: number };
  sampleWeight?: tf.Tensor;
  initialEpoch?: number;
  stepsPerEpoch?: number;
  steps?: number;
  validationSteps?: number;
  metrics?: string[] | {[outputName: string]: string};
  verbose?: ModelLoggingVerbosity;
  handlerOrURL?: string;
};

export interface IConfigurationParams extends Partial<IParams> {
  optimizer?: string | tf.Optimizer;
  loss?: string | string[];
  callbacks?: IParamsCallbacks;
}
