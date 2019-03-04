/* globals Promise */
import * as tf from '@tensorflow/tfjs';
import MLClassifier from 'ml-classifier';
import dog from './images/dog.jpg';
import cat from './images/cat.png';

const mlClassifier = new MLClassifier();

const loadImage = (src) => new Promise((resolve, reject) => {
  const image = new Image();
  image.src = src;
  image.crossOrigin = '';
  image.onload = () => resolve(image);
  image.onerror = (err) => reject(err);
});

function cropImage(img) {
  const height = img.shape[0];
  const width = img.shape[1];

  // use the shorter side as the size to which we will crop
  const shorterSide = Math.min(img.shape[0], img.shape[1]);

  // calculate beginning and ending crop points
  const startingWidth = Math.floor((width - shorterSide) / 2);
  const startingHeight = Math.floor((height - shorterSide) / 2);
  const endingWidth = Math.floor(startingWidth + shorterSide);
  const endingHeight = Math.floor(startingHeight + shorterSide);

  // return image data cropped to those points
  return img.slice([startingHeight, startingWidth, 0], [endingHeight, endingWidth, 3]);
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

const parseImage = async (src) => {
  const img = await loadImage(src);
  const pixels = tf.browser.fromPixels(img);
  const imageData = loadAndProcessImage(pixels);
  return imageData;
};

(async function() {
  // const dogPixels = await parseImage(dog);
  // const catPixels = await parseImage(cat);
  // const images = dogPixels.concat(catPixels);

  const images = [dog, cat];
  const labels = ['dog', 'cat'];
  await mlClassifier.addData(images, labels, 'train');
  mlClassifier.train({
    callbacks: {
      onTrainBegin: () => {
        console.log('training begins');
      },
      onTrainEnd: () => {
        console.log('training ends');
      },
    },
  });
})();

