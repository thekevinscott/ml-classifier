# ML Classifier

ML Classifier is a machine learning engine for quickly training image classification models in your browser. Models can be saved with a single command, and the resulting models reused to make image classification predictions.

This package is intended as a companion for [`ml-classifier-ui`](https://github.com/thekevinscott/ml-classifier-ui), which provides a web frontend in React for uploading data and seeing results.

## Walkthrough

A walkthrough of the code can be found in the article [Image Classification in the Browser with Javascript](https://thekevinscott.com/image-classification-with-javascript/).

## Demo

An interactive [demo can be found here](https://thekevinscott.github.io/ml-classifier-ui/).

![Demo](https://github.com/thekevinscott/ml-classifier-ui/raw/master/example/public/example.gif)
*Screenshot of demo*

## Getting Started

### Installation

`ml-classifier` can be installed via `yarn` or `npm`:

```
yarn add ml-classifier
```

or

```
npm install ml-classifier
```

### Quick Start

Start by instantiating a new MLClassifier.

```
import MLClassifier from 'ml-classifier';

const mlClassifier = new MLClassifier();
```

Then, train the model:

```
await mlClassifier.train(imageData, {
  callbacks: {
    onTrainBegin: () => {
      console.log('training begins');
    },
    onBatchEnd: (batch: any,logs: any) => {
      console.log('Loss is: ' + logs.loss.toFixed(5));
    }
  },
});
```

And get predictions:

```
const prediction = await mlClassifier.predict(data);
```

When you have a trained model you're happy with, save it with:

```
mlClassifier.save();
```

## Using the saved model

When you hit save, Tensorflow.js will download a weights file and a model topology file.

You'll need to combine both into a single `json` file. Open up your model topology file and at the top level of the JSON file, make sure to add a `weightsManifest` key pointing to your weights, like:

```
{
  "weightsManifest": "ml-classifier-class1-class2.weights.bin",
  "modelTopology": {
    ...
  }
}
```

When using the model in your app, there's a few things to keep in mind:

1. You need to make sure you transform images into the correct dimensions, depending on the pretrained model it was trained with. (For MOBILENET, this would be 1x224x224x3).
2. You must create a pretrained model matching the dimensions used to train. An example is below for MOBILENET.
3. You must first run your images through the pretrained model to activate them.
4. After getting the final prediction, you must take the arg max.
5. You'll get back a number indicating your class.

Full example for MOBILENET:

```
    const loadImage = (src) => new Promise((resolve, reject) => {
      const image = new Image();
      image.src = src;
      image.crossOrigin = 'Anonymous';
      image.onload = () => resolve(image);
      image.onerror = (err) => reject(err);
    });

    const pretrainedModelURL = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';

    tf.loadModel(pretrainedModelURL).then(model => {
      const layer = model.getLayer('conv_pw_13_relu');
      return tf.model({
        inputs: [model.inputs[0]],
        outputs: layer.output,
      });
    }).then(pretrainedModel => {
      return tf.loadModel('/model.json').then(model => {
        return loadImage('/trees/tree1.png').then(loadedImage => {
          const image = tf.reshape(tf.fromPixels(loadedImage), [1,224,224,3]);
          const pretrainedModelPrediction = pretrainedModel.predict(image);
          const modelPrediction = model.predict(pretrainedModelPrediction);
          const prediction = modelPrediction.as1D().argMax().dataSync()[0];
          console.log(prediction);
        });
      });
    }).catch(err => {
      console.error('Error', err);
    });
```

## API Documentation

Start by instantiating a new instance of `MLClassifier` with:

```
const mlClassifier = new MLClassifier();
```

This will begin loading the pretrained model and provide you with an object onto which to add data and train.

### `constructor`

`MLClassifier` accepts a number of callbacks for beginning and end of various methods.

You can provide a custom pretrained model as a `pretrainedModel`.

You can provide a custom training model as a `trainingModel`.

#### Parameters

  * **pretrainedModel** (`string | tf.Model`) *Optional* - A string denoting which pretrained model to load from an internal config. Valid strings can be found on the exported object `PRETRAINED_MODELS`. You can also specify a preloaded pretrained model directly.
  * **trainingModel** (`tf.Model | Function`) *Optional* - A custom model to use during training. Can be provided as a `tf.Model` or as a function that accepts `{xs: [...], ys: [...]`, number of `classes`, and `params` provided to train.
  * **onLoadStart** (`Function`) *Optional* - A callback for when `load` (loading the pre-trained model) is first called.
  * **onLoadComplete** (`Function`) *Optional* - A callback for when `load` (loading the pre-trained model) is complete.
  * **onAddDataStart** (`Function`) *Optional* - A callback for when `addData` is first called.
  * **onAddDataComplete** (`Function`) *Optional* - A callback for when `addData` is complete.
  * **onClearDataStart** (`Function`) *Optional* - A callback for when `clearData` is first called.
  * **onClearDataComplete** (`Function`) *Optional* - A callback for when `clearData` is complete.
  * **onTrainStart** (`Function`) *Optional* - A callback for when `train` is first called.
  * **onTrainComplete** (`Function`) *Optional* - A callback for when `train` is complete.
  * **onEvaluateStart** (`Function`) *Optional* - A callback for when `evaluate` is first called.
  * **onEvaluateComplete** (`Function`) *Optional* - A callback for when `evaluate` is complete.
  * **onPredictStart** (`Function`) *Optional* - A callback for when `predict` is first called.
  * **onPredictComplete** (`Function`) *Optional* - A callback for when `predict` is complete.
  * **onSaveStart** (`Function`) *Optional* - A callback for when `save` is first called.
  * **onSaveComplete** (`Function`) *Optional* - A callback for when `save` is complete.


#### Example
```
import MLClassifier, {
  PRETRAINED_MODELS,
} from 'ml-classifier';

const mlClassifier = new MLClassifier({
  pretrainedModel: PRETRAINED_MODELS.MOBILENET,

  onLoadStart: () => console.log('onLoadStart'),
  onLoadComplete: () => console.log('onLoadComplete'),
  onAddDataStart: () => console.log('onAddDataStart'),
  onAddDataComplete: () => console.log('onAddDataComplete'),
  onClearDataStart: () => console.log('onClearDataStart'),
  onClearDataComplete: () => console.log('onClearDataComplete'),
  onTrainStart: () => console.log('onTrainStart'),
  onTrainComplete: () => console.log('onTrainComplete'),
  onEvaluateStart: () => console.log('onEvaluateStart'),
  onEvaluateComplete: () => console.log('onEvaluateComplete'),
  onPredictStart: () => console.log('onPredictStart'),
  onPredictComplete: () => console.log('onPredictComplete'),
  onSaveStart: () => console.log('onSaveStart'),
  onSaveComplete: () => console.log('onSaveComplete'),
});
```

Example of specifying a preloaded pretrained model:

```
import MLClassifier from 'ml-classifier';

const mlClassifier = tf.loadModel('... some pretrained model ...').then(model => {
  return new MLClassifier({
    pretrainedModel: model,
  });
});
```

### `addData`

This method takes an array of incoming images, an optional array of labels, and an optional dataType.

#### Example

```
import MLClassifier from 'ml-classifier';
const mlClassifier = new MLClassifier();
mlClassifier.addData(images, labels, 'train');
```

#### Parameters

* **images** (`Array<tf.Tensor3D | ImageData | HTMLImageElement | string>`) - an array of 3D tensors, ImageData (output from a canvas `toPixels`, a native browser `Image`, or a string representing the image `src`. Images can be any sizes, but will be cropped and sized down to match the pretrained model.
* **labels** (`string[]`) - an array of strings, matching the images passed above.
* **dataType** (`string`) *Optional* - an enum specifying which data type the images match. Data types can be `train` for data used in `model.train()`, and `eval`, for data used in `model.evaluate()`. If no argument is supplied, `dataType` will default to `train`.

#### Returns

Nothing.

### `train`

`train` begins training on the given dataset.

#### Example

```
import MLClassifier from 'ml-classifier';
const mlClassifier = new MLClassifier();
mlClassifier.addData(images, labels, DataType.TRAIN);
mlClassifier.train({
  callbacks: {
    onTrainBegin: () => {
      console.log('training begins');
    },
  },
});
```

#### Parameters

* **params** (`Object`) *Optional* - a set of parameters that will be passed directly to `model.fit`. [View the Tensorflow.JS docs](https://js.tensorflow.org/api/0.12.0/#tf.Model.fit) for an up-to-date list of arguments.

#### Returns

`train` returns the resolved promise from `fit`, an object containing loss and accuracy.

## `evaluate`

`evaluate` is used to evaluate a model's performance.

#### Example

```
import MLClassifier from 'ml-classifier';
const mlClassifier = new MLClassifier();
mlClassifier.addData(images, labels, DataType.TRAIN);
mlClassifier.train();
mlClassifier.addData(evaluationImages, labels, DataType.EVALUATE);
mlClassifier.evaluate();
```

#### Parameters

* **params** (`Object`) *Optional* - a set of parameters that will be passed directly to `model.evaluate`. [View the Tensorflow.JS docs](https://js.tensorflow.org/api/0.12.0/#tf.Sequential.evaluate) for an up-to-date list of arguments.

#### Returns

`evaluate` returns a tf.Scalar representing the result of `evaluate`.

## `predict`

`predict` is used to make a specific prediction using a saved model.

#### Example

```
import MLClassifier from 'ml-classifier';
const mlClassifier = new MLClassifier();
mlClassifier.addData(images, labels, DataType.TRAIN);
mlClassifier.train();
mlClassifier.predict(imageToPredict);
```

#### Parameters

* **image** (`tf.Tensor3D`) - a single image encoded as a `tf.Tensor3D`. Image can be any size, but will be cropped and sized down to match the pretrained model.

#### Returns

`predict` will return a string matching the prediction.

## `save`

`save` is a proxy to `tf.model.save`, and will initiate a download from the browser, or save to local storage.

#### Example

```
import MLClassifier from 'ml-classifier';
const mlClassifier = new MLClassifier();
mlClassifier.addData(images, labels, DataType.TRAIN);
mlClassifier.train();
mlClassifier.save(('path-to-save');
```

#### Parameters

* **handlerOrUrl** (`io.IOHandler | string`) *Optional* - an argument to be passed to `model.save`. If omitted, the model's unique labels will be concatenated together in the form of `class1-class2-class3`.
* **params** (`Object`) *Optional* - a set of parameters that will be passed directly to `model.save`. [View the Tensorflow.JS docs](https://js.tensorflow.org/api/0.12.0/#tf.Model.save) for an up-to-date list of arguments.


## `getModel`

`getModel` will return the trained Tensorflow.js model. Calling this method prior to calling `mlClassifier.train` will return `null`.

#### Example

```
import MLClassifier from 'ml-classifier';
const mlClassifier = new MLClassifier();
mlClassifier.addData(images, labels, DataType.TRAIN);
mlClassifier.train();
mlClassifier.getModel();
```

#### Parameters

None.

#### Returns

The saved Tensorflow.js model.

## `clearData`

`clearData` will clear out saved data.

#### Example
```
import MLClassifier from 'ml-classifier';
const mlClassifier = new MLClassifier();
mlClassifier.addData(images, labels, DataType.TRAIN);
mlClassifier.clearData(DataType.TRAIN);
```

#### Parameters

* **dataType** (`DataType`) *Optional* - specifies which data to clear. If no argument is provided, all data will be cleared.

#### Returns

Nothing.

## Contributing

Contributions are welcome!

You can start up a local copy of `ml-classifier` with:

```
yarn watch
```

`ml-classifier` is written in Typescript.

### Tests

Tests are a work in progress. Currently, the test suite only consists of unit tests. Pull requests for additional tests are welcome!

Run tests with:

```
yarn test
```

## Author

* [Kevin Scott](https://thekevinscott.com)

## License

This project is licensed under the MIT License - see the LICENSE file for details

![](https://ga-beacon.appspot.com/UA-112845439-4/ml-classifier/readme)
