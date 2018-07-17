import * as tf from '@tensorflow/tfjs';

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
  console.log('prepare to get arr');
  const arr = await imageToUint8ClampedArray(image, dims);
  console.log('got arr, prepare to get tensor');
  return imageDataToTensor(arr);
}

const imageDataToTensor = async ({
  data,
  width,
  height,
}: IImageData) => {
  console.log('prepare to get tensor', width, height);
  return tf.tensor3d(Array.from(data), [width, height, 4]);
};

interface IImageData {
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

export interface ImageError {
  image: any;
  error: Error,
}

const getTranslatedImageAsTensor = async (image: tf.Tensor3D | IImageData | HTMLImageElement | string, dims: [number, number]) => {
  if (image instanceof tf.tensor3d) {
    console.log('tensor');
    return image;
  } else if (typeof image === 'string') {
    console.log('strgin');
    const loadedImage = await loadImage(image);
    console.log('got loaded image');
    return await loadTensorFromHTMLImage(loadedImage, dims);
  } else if (image instanceof HTMLImageElement) {
    console.log('html image el');
    return await loadTensorFromHTMLImage(image, dims);
  } else if (image instanceof ImageData) {
    console.log('image data');
    return await imageDataToTensor(image);
  }

  throw new Error('Unsupported image type');
};

const translateImages = async (origImages: Array<tf.Tensor3D | IImageData | HTMLImageElement | string>, dims: [number, number], origLabels?: string[]) => {
  const images = [];
  const errors: ImageError[] = [];
  const labels = [];

  for (let i = 0; i < origImages.length; i++) {
    console.log('translating', i);
    const origImage = origImages[i];
    try {
      console.log('prepare to get');
      const image = await getTranslatedImageAsTensor(origImage, dims);
      console.log('gotten');
      // else, it is already a tensor

      images.push(image);
      if (origLabels) {
        labels.push(origLabels[i]);
      }
    } catch(error) {
      console.log('error for', i);
      errors.push({
        image: origImage,
        error,
      });
    }
  }

  return {
    images,
    errors,
    labels,
  };
}

export default translateImages;
