# ML Classifier

# Installation

```
yarn add create-ml
```

or

```
npm install create-ml
```

# Usage

```
import createML from 'create-ml';

const ml = createML({
  ...params,
});

ml.train(images, {
  ...params,
});

ml.predict(images, {
  callback: (image, prediction => {
  });
});

ml.download();

ml.getModel().print();
```

You instantiate `createML` by (optionally) passing it an object of parameters.

```
{
  epochs: 20,
  batchSize: 10,
}
```

## Train
Train accepts the same kinds of parameters:

```
ml.train(images, {
  epochs: 20,
  batchSize: 10,
});
```

Any parameters provided will overwrite the initialized parameters.

```
images = [{
  data: pixel data,
  label: 'strawberry',
}, {
  data: pixel data,
  label: 'blueberry',
}];
```

`train` returns a promise indicating completion of the training.

*This should support all the fit callbacks*
https://js.tensorflow.org/api/0.11.7/#tf.Model.fit

`train` is chainable.

## Predict

```
ml.predict(image, {
});
```

ml.predict accepts a single pixel data array, and returns a single class prediction.

## Download

```
ml.download();
```

`download` is chainable

This will initiate a download from the browser
