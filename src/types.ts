import * as tf from '@tensorflow/tfjs';

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
  xs: tf.Tensor3D | null;
  ys: tf.Tensor2D | null;
}

export interface IParamsCallbacks {
  onTrainBegin?: Function;
  onTrainEnd?: Function;
  onEpochBegin?: Function;
  onEpochEnd?: Function;
  onBatchBegin?: Function;
  onBatchEnd?: Function;
};

export interface IConfigurationParams {
  // optimizer?: tf.train.Optimizer;
  optimizer?: any;
  // loss?: string | string[] | tf.LossOrMetricFn;
  loss?: string | string[];
  layers?: Function;
  // layers?: ({
  //   classes: number;
  //   xs: tf.Tensor3D | null;
  //   ys: tf.Tensor2D | null;
  // }) => tf.layers.Layer[];
  model?: tf.Model;
  batchSize?: number;
  epochs?: number;
  callbacks?: IParamsCallbacks;
  validationSplit?: number;
  validationData?: [ tf.Tensor|tf.Tensor[], tf.Tensor|tf.Tensor[] ]|[tf.Tensor | tf.Tensor[], tf.Tensor|tf.Tensor[], tf.Tensor|tf.Tensor[]];
  shuffle?: boolean;
  classWeight?: {[classIndex: string]: number };
  sampleWeight?: tf.Tensor;
  initialEpoch?: number;
  stepsPerEpoch?: number;
  validationSteps?: number;
  metrics?: string[] | {[outputName: string]: string};
  verbose?: boolean;
}

export interface IParams {
  // optimizer?: tf.train.Optimizer;
  optimizer: any;
  // loss?: string | string[] | tf.LossOrMetricFn;
  loss: string | string[];
  layers?: Function;
  // layers?: ({
  //   classes: number;
  //   xs: tf.Tensor3D | null;
  //   ys: tf.Tensor2D | null;
  // }) => tf.layers.Layer[];
  model?: tf.Model;
  batchSize?: number;
  epochs?: number;
  callbacks?: IParamsCallbacks;
  validationSplit?: number;
  validationData?: [ tf.Tensor|tf.Tensor[], tf.Tensor|tf.Tensor[] ]|[tf.Tensor | tf.Tensor[], tf.Tensor|tf.Tensor[], tf.Tensor|tf.Tensor[]];
  shuffle?: boolean;
  classWeight?: {[classIndex: string]: number };
  sampleWeight?: tf.Tensor;
  initialEpoch?: number;
  stepsPerEpoch?: number;
  validationSteps?: number;
  metrics?: string[] | {[outputName: string]: string};
  verbose?: boolean;
  handlerOrURL?: string;
};
