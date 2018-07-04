import {
  ITrainingData,
} from './types';

const Haikunator = require('haikunator');

const haikunator = new Haikunator();

const getName = (data: ITrainingData) => {
  const classes = Object.keys(data.classes);
  if (classes.length < 4) {
    return classes.join('-');
  }

  return haikunator.haikunate();
};

const getDefaultDownloadHandler = (data: ITrainingData) => {
  return `downloads://ml-classifier-${getName(data)}`;
};

export default getDefaultDownloadHandler;
