import * as tf from '@tensorflow/tfjs';

const crop = (img: tf.Tensor3D) => {
  const size = Math.min(img.shape[0], img.shape[1]);
  const centerHeight = img.shape[0] / 2;
  const beginHeight = centerHeight - (size / 2);
  const centerWidth = img.shape[1] / 2;
  const beginWidth = centerWidth - (size / 2);
  return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
}

// convert pixel data into a tensor
const prepareData = async (img: tf.Tensor3D): Promise<tf.Tensor3D> => {
  return tf.tidy(() => {
    console.log('img', img);
    const croppedImage = crop(tf.image.resizeBilinear(img, [224, 224]));
    const batchedImage = croppedImage.expandDims(0);
    return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
  });
};

export default prepareData;
