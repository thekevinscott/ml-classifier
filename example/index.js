/* globals Promise */
// import * as tf from '@tensorflow/tfjs';
import dog from './dog.jpg';
import MLClassifier, { tf2 as tf } from '../dist';
console.log(MLClassifier);
// const mlClassifier = new MLClassifier();

const loadImage = (src) => new Promise((resolve, reject) => {
  const image = new Image();
  image.src = src;
  image.crossOrigin = '';
  image.onload = () => resolve(image);
  image.onerror = (err) => reject(err);
});

function cropImage(img) {
  const width = img.shape[0];
  const height = img.shape[1];

  // use the shorter side as the size to which we will crop
  const shorterSide = Math.min(img.shape[0], img.shape[1]);

  // calculate beginning and ending crop points
  const startingHeight = (height - shorterSide) / 2;
  const startingWidth = (width - shorterSide) / 2;
  const endingHeight = startingHeight + shorterSide;
  const endingWidth = startingWidth + shorterSide;

  // return image data cropped to those points
  return img.slice([startingWidth, startingHeight, 0], [endingWidth, endingHeight, 3]);
}
function resizeImage(image) {
  return tf.image.resizeBilinear(image, [224, 224]);
}
function batchImage(image) {
  // Expand our tensor to have an additional dimension, whose size is 1
  const batchedImage = image.expandDims(0);

  // Turn pixel data into a float between -1 and 1.
  return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
}
function loadAndProcessImage(image) {
  const croppedImage = cropImage(image);
  const resizedImage = resizeImage(croppedImage);
  const batchedImage = batchImage(resizedImage);
  return batchedImage;
}

loadImage(dog).then(img => {
  console.log('tf', tf);
  const pixels = tf.browser.fromPixels(img);
  const imageData = loadAndProcessImage(pixels);
  // mlClassifier.addData(images, labels, DataType.TRAIN);
  // mlClassifier.train({
  //   callbacks: {
  //     onTrainBegin: () => {
  //       console.log('training begins');
  //     },
  //   },
  // });
});
