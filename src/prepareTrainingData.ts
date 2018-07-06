import * as tf from '@tensorflow/tfjs';
import getClasses from './getClasses';

import {
  ITrainingData,
  IActivatedImage,
} from './types';

const oneHot = (labelIndex: number, classLength: number) => tf.tidy(() => tf.oneHot(tf.tensor1d([labelIndex]).toInt(), classLength));

const prepareTrainingData = (images: IActivatedImage[]): ITrainingData => {
  return images.reduce((data: ITrainingData, { activation, label }: IActivatedImage) => {
    const labelIndex = data.classes[label];
    const classLength = Object.keys(data.classes).length;
    const y = oneHot(labelIndex, classLength);

    return tf.tidy(() => {
      if (data.xs === undefined || data.ys === undefined) {
        return {
          ...data,
          xs: tf.keep(activation),
          ys: tf.keep(y),
        };
      }

      const oldX = data.xs;
      const oldY = data.ys;

      const xs = tf.keep(oldX.concat(activation, 0));
      const ys = tf.keep(oldY.concat(y, 0));

      oldX.dispose();
      oldY.dispose();
      y.dispose();

      return {
        ...data,
        xs,
        ys,
      };
    });
  }, {
    classes: getClasses(images),
  });
};

export default prepareTrainingData;
