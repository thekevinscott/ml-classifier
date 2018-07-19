import * as tf from '@tensorflow/tfjs';

import {
  IClasses,
} from './types';

const oneHot = (labelIndex: number, classLength: number) => tf.tidy(() => tf.oneHot(tf.tensor1d([labelIndex]).toInt(), classLength));

// const turnTensorArrayIntoTensor = (tensors: tf.Tensor[]) => tensors.reduce((data?: tf.Tensor, tensor: tf.Tensor) => tf.tidy(() => {
//   if (data === undefined) {
//     return tf.keep(tensor);
//   }

//   const newData = tf.keep(data.concat(tensor, 0));
//   data.dispose();
//   return newData;
// }), undefined);

export const addData = (tensors: tf.Tensor[]): tf.Tensor => {
  const data = tf.keep(tensors[0]);
  return tensors.slice(1).reduce((data: tf.Tensor, tensor: tf.Tensor) => tf.tidy(() => {
    const newData = tf.keep(data.concat(tensor, 0));
    data.dispose();
    return newData;
  }), data);
};

export const addLabels = (labels: string[], classes: IClasses): tf.Tensor2D | undefined => {
  const classLength = Object.keys(classes).length;
  if (classLength <= 1) {
    throw new Error('You must provide more than 1 class for training');
  }

  return labels.reduce((data: tf.Tensor2D | undefined, label: string) => {
    const labelIndex = classes[label];
    const y = oneHot(labelIndex, classLength);

    return tf.tidy(() => {
      if (data === undefined) {
        return tf.keep(y);
      }

      const old = data;
      const ys = tf.keep(old.concat(y, 0));

      old.dispose();
      y.dispose();

      return ys;
    });
  }, undefined);
};
