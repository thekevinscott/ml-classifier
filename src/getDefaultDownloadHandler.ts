import {
  IClasses,
} from './types';

const Haikunator = require('haikunator');

const haikunator = new Haikunator();

const getOrderedClasses = (classes: IClasses) => Object.entries(classes).sort((a, b) => {
  return a[1] - b[1];
}).map(([key]) => key);

export const getName = (classes: IClasses) => {
  const orderedClasses = getOrderedClasses(classes);
  if (orderedClasses.length < 4) {
    return orderedClasses.join('-');
  }

  return haikunator.haikunate();
};

const getDefaultDownloadHandler = (classes: IClasses) => {
  return `downloads://ml-classifier-${getName(classes)}`;
};

export default getDefaultDownloadHandler;
