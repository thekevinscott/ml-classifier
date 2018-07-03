import {
  IImage,
  IActivatedImage,
} from './types';

const getClasses = (images: (IImage | IActivatedImage)[]) => images.reduce((labels, { label }) => {
  if (labels[label] !== undefined) {
    return labels;
  }

  return {
    ...labels,
    [label]: Object.keys(labels).length,
  };
}, {});

export default getClasses;
