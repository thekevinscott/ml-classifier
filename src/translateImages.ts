import * as tf from '@tensorflow/tfjs';

const loadImage = async (src: string) => new Promise<HTMLImageElement>((resolve, reject) => {
  const image = new Image();
  image.src = src;
  image.onload = () => resolve(image);
  image.onerror = (err) => reject(err);
});

const translateImages = async (origImages: Array<tf.Tensor3D | HTMLImageElement | string>) => {
  const images = [];

  for (let i = 0; i < origImages.length; i++) {
    let image = origImages[i];

    if (typeof image === 'string') {
      const loadedImage = await loadImage(image);
      image = tf.fromPixels(loadedImage);
    } else if (image instanceof HTMLImageElement) {
      const loadedImage = await loadImage(image.src);
      image = tf.fromPixels(loadedImage);
    }
    // else, it is a tensor

    images.push(image);
  }

  return images;
}

export default translateImages;
