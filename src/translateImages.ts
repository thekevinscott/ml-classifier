import * as tf from '@tensorflow/tfjs';
// import log from './log';

// import {
//   TypedArray,
// } from './types';

const loadImage = async (src: string) => new Promise<HTMLImageElement>((resolve, reject) => {
  const image = new Image();
  image.src = src;
  image.crossOrigin = 'Anonymous';
  image.onload = () => resolve(image);
  image.onerror = (err) => reject(err);
});

const imageToUint8ClampedArray = async (image: HTMLImageElement, dims: [number, number]) => {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  if (context) {
    context.drawImage(image, 0, 0, image.width, image.height, 0, 0, ...dims);
    const data = context.getImageData(0, 0, ...dims)
    // const data = context.getImageData(0, 0, image.width, image.height);
    return data;
  }

  throw new Error('No context found; are you in the browser?');
};

const loadTensorFromHTMLImage = async (image: HTMLImageElement, dims: [number, number]) => {
  const arr = await imageToUint8ClampedArray(image, dims);
  return imageDataToTensor(arr);
}

const imageDataToTensor = async ({
  data,
  width,
  height,
}: IImageData) => {
  return tf.tensor3d(Array.from(data), [width, height, 4]);
};

export interface IImageData {
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

export interface ImageError {
  image: any;
  error: Error;
  index: number;
}

const getTranslatedImageAsTensor = async (image: tf.Tensor | IImageData | HTMLImageElement | string, dims: [number, number], num: number) => {
  await tf.nextFrame();
  if (image instanceof tf.tensor3d) {
    return image;
  } else if (typeof image === 'string') {
    const loadedImage = await loadImage(image);
    await tf.nextFrame();
    return await loadTensorFromHTMLImage(loadedImage, dims);
  } else if (image instanceof HTMLImageElement) {
    const tensor = await loadTensorFromHTMLImage(image, dims);
    await tf.nextFrame();
    return tensor;
  } else if (image instanceof ImageData) {
    return await imageDataToTensor(image);
  }

  throw new Error('Unsupported image type');
};

const translateImages = async (origImages: Array<tf.Tensor | IImageData | HTMLImageElement | string>, dims: [number, number], origLabels?: string[]) => {
  const images = [];
  const errors: ImageError[] = [];
  const labels = [];

  for (let i = 0; i < origImages.length; i++) {
    const origImage = origImages[i];
    try {
      const image = await getTranslatedImageAsTensor(origImage, dims, i);
      // else, it is already a tensor

      images.push(image);
      if (origLabels) {
        labels.push(origLabels[i]);
      }
    } catch(error) {
      errors.push({
        image: origImage,
        error,
        index: i,
      });
    }
    await tf.nextFrame();
  }

  return {
    images,
    errors,
    labels,
  };
}

export default translateImages;
