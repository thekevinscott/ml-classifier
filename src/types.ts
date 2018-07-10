import * as tf from '@tensorflow/tfjs';

export interface IClasses {
  [index: string]: number;
}

export enum DataType {
  TRAIN = "train",
  EVAL = "eval",
};
export interface IData {
  classes: IClasses;
  [index: string]: IImageData;
}

export interface IImageData {
  xs?: tf.Tensor3D;
  ys?: tf.Tensor2D;
}

export interface ICollectedData {
  classes: IClasses;
  xs?: tf.Tensor3D;
  ys?: tf.Tensor2D;
}

export interface IParams {
  [index: string]: any;
  batchSize?: number;
  epochs?: number;
};
