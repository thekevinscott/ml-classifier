import {
  ITrainingData,
  IClasses,
} from './types';

const Haikunator = require('haikunator');

const haikunator = new Haikunator();

const getOrderedClasses = (classes: IClasses) => Object.entries(classes).sort((a, b) => {
  return a[1] - b[1];
}).map(([key]) => key);

export const getName = (data: ITrainingData) => {
  const classes = getOrderedClasses(data.classes);
  if (classes.length < 4) {
    return classes.join('-');
  }

  return haikunator.haikunate();
};

const getDefaultDownloadHandler = (data: ITrainingData) => {
  return `downloads://ml-classifier-${getName(data)}`;
};

export default getDefaultDownloadHandler;
