import * as tf from '@tensorflow/tfjs';

export const PRETRAINED_MODELS_KEYS = {
  MOBILENET: 'mobilenet_v1_0.25_224',
}

export const PRETRAINED_MODELS = {
  [PRETRAINED_MODELS_KEYS.MOBILENET]: {
    url: 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json',
    layer: 'conv_pw_13_relu',
  },
};

const loadPretrainedModel = async (pretrainedModel: string | tf.Model = PRETRAINED_MODELS_KEYS.MOBILENET) => {
  if (pretrainedModel instanceof tf.Model) {
    return pretrainedModel;
  }

  if (!PRETRAINED_MODELS[pretrainedModel]) {
    throw new Error('You have supplied an invalid key for a pretrained model');
  }

  const config = PRETRAINED_MODELS[pretrainedModel];
  const model = await tf.loadModel(config.url);
  const layer = model.getLayer(config.layer);
  return tf.model({
    inputs: [model.inputs[0]],
    outputs: layer.output,
  });
};

export default loadPretrainedModel;
