import * as tf from '@tensorflow/tfjs';

export interface IClasses {
  [index: string]: number;
}

// export enum DataType {
//   TRAIN = "train",
//   EVAL = "eval",
// };
export interface IData {
  classes: IClasses;
  [index: string]: IImageData;
}

export interface IImageData {
  xs?: tf.Tensor;
  ys?: tf.Tensor2D;
}

export interface ICollectedData {
  classes: IClasses;
  xs?: tf.Tensor;
  ys?: tf.Tensor2D;
}

export interface IParams {
  [index: string]: any;
  batchSize?: number;
  epochs?: number;
};

export type TypedArray = Int8Array | Uint8Array | Int16Array | Uint16Array | Int32Array | Uint32Array | Uint8ClampedArray | Float32Array | Float64Array;

export interface IArgs {
  pretrainedModel?: string | tf.LayersModel;
  trainingModel?: tf.LayersModel | Function;
  // trainingModel?: tf.Model | (data: IImageData, classes: number, params: IParams) => tf.Model;

  onLoadStart?: Function;
  onLoadComplete?: Function;
  onAddDataStart?: Function;
  onAddDataComplete?: Function;
  onClearDataStart?: Function;
  onClearDataComplete?: Function;
  onTrainStart?: Function;
  onTrainComplete?: Function;
  onPredictComplete?: Function;
  onPredictStart?: Function;
  onEvaluateStart?: Function;
  onEvaluateComplete?: Function;
  onSaveStart?: Function;
  onSaveComplete?: Function;
}

