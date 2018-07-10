import * as tf from '@tensorflow/tfjs';
import getDefaultDownloadHandler from './getDefaultDownloadHandler';
jest.mock('./getDefaultDownloadHandler');
jest.genMockFromModule('@tensorflow/tfjs');
jest.mock('@tensorflow/tfjs', () => ({
  train: {
    adam: () => {},
  },
  model: ({
    save: (handlerOrURL) => {
      return handlerOrURL;
    },
  }),
  loadModel: () => ({
    getLayer: () => ({
      output: null,
    }),
    inputs: [],
  }),
}));
import MLClassifier from './index';

describe('ml-classifier', () => {
  test('foo', () => {
    expect('a').toEqual('a');
  });
  // describe('constructor', () => {
  //   test('that it persists params', async () => {
  //     const epochs = 123;
  //     const mlClassifier = new MLClassifier({
  //       epochs,
  //     });
  //     expect(mlClassifier.getParams().epochs).toEqual(epochs);
  //   });

  //   // test('that it calls init on construct', async () => {
  //   //   MLClassifier.prototype.init = jest.fn(() => {});
  //   //   const mlClassifier = new MLClassifier({ });
  //   //   expect(mlClassifier.init).toHaveBeenCalled();
  //   // });
  // });

  // describe('save', () => {
  //   let mlClassifier;
  //   beforeEach(() => {
  //     mlClassifier = new MLClassifier();
  //     mlClassifier.loaded = jest.fn(() => {});
  //   });

  //   test('it waits for pretrained model as the first step', async () => {
  //     mlClassifier.model = tf.model;
  //     await mlClassifier.save();
  //     expect(mlClassifier.loaded).toHaveBeenCalled();
  //   });

  //   test('it throws if no model is set', async () => {
  //     const expectedError = new Error('You must call train prior to calling save');
  //     return mlClassifier.save().catch(err => {
  //       expect(err.message).toBe(expectedError.message);
  //     });
  //   });

  //   test('calls save with a handler if specified', async () => {
  //     const url = 'foobar';
  //     mlClassifier.model = tf.model;
  //     const result = await mlClassifier.save(url);
  //     expect(result).toEqual(url);
  //   });

  //   test('calls save with a default handler if none is specified', async () => {
  //     const def = 'def';
  //     getDefaultDownloadHandler.mockImplementationOnce(() => def);
  //     mlClassifier.model = tf.model;
  //     const result = await mlClassifier.save();
  //     expect(result).toEqual(def);
  //   });
  // });
});
