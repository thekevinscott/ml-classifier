import * as tf from '@tensorflow/tfjs';
import loadPretrainedModel, {
  PRETRAINED_MODELS_KEYS,
  PRETRAINED_MODELS,
} from './loadPretrainedModel';

jest.genMockFromModule('@tensorflow/tfjs');
jest.mock('@tensorflow/tfjs', () => ({
  model: (params) => ({
    ...params,
    save: (handlerOrURL) => {
      return handlerOrURL;
    },
  }),
  loadModel: jest.fn((url) => ({
    getLayer: () => ({
      output: null,
    }),
    inputs: [],
  })),
}));

describe('loadPretrainedModel', () => {
  test('it throws an error if an invalid key is provided', async () => {
    return loadPretrainedModel('foo').catch(err => {
      expect(err.message).toEqual('You have supplied an invalid key for a pretrained model');
    });
  });

  test('loads a pretrained model specified in the config with tf.loadModel', async () => {
    const loadModel = jest.spyOn(tf, 'loadModel');
    const model = await loadPretrainedModel(PRETRAINED_MODELS_KEYS.MOBILENET);
    expect(loadModel).toHaveBeenCalledWith(PRETRAINED_MODELS[PRETRAINED_MODELS_KEYS.MOBILENET].url);
  });
});
