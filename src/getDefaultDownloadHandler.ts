import {
  IClasses,
} from './types';

const getOrderedClasses = (classes: IClasses) => Object.entries(classes).sort((a, b) => {
  return a[1] - b[1];
}).map(([key]) => key);

export const getName = (classes: IClasses) => getOrderedClasses(classes).join('-');

const getDefaultDownloadHandler = (classes: IClasses) => {
  return `downloads://ml-classifier-${getName(classes)}`;
};

export default getDefaultDownloadHandler;
