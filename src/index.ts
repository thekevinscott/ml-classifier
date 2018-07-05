import * as tf from '@tensorflow/tfjs';
import prepareData from './prepareData';
import train from './train';
import loadPretrainedModel, {
  PRETRAINED_MODELS_KEYS,
} from './loadPretrainedModel';
import prepareTrainingData from './prepareTrainingData';
import getDefaultDownloadHandler from './getDefaultDownloadHandler';

import {
  IImage,
  IConfigurationParams,
  IParams,
  ITrainingData,
} from './types';

const defaultParams = {
  epochs: 20,
  loss: 'categoricalCrossentropy',
  optimizer: tf.train.adam(0.0001),
  callbacks: {},
};

class MLClassifier {
  private params: IParams;
  // private pretrainedModel: typeof tf.model;
  private pretrainedModel: any;
  private model: tf.Sequential;
  private callbacks: Function[] = [];
  private data: ITrainingData;
  public tf = tf;

  constructor(params: IConfigurationParams = {}) {
    this.params = {
      ...defaultParams,
      ...params,
    };

    this.init();
  }

  private init = async () => {
    this.pretrainedModel = await loadPretrainedModel(PRETRAINED_MODELS_KEYS.MOBILENET);

    this.callbacks.map(callback => callback());
  }

  public getModel = () => this.model;

  private loaded = async () => new Promise(resolve => {
    if (this.pretrainedModel) {
      return resolve();
    }

    this.callbacks.push(() => {
      resolve();
    });
  });

  private prepareData = async (image: tf.Tensor3D) => {
    await this.loaded();
    const processedImage = await prepareData(image);
    return this.pretrainedModel.predict(processedImage);
  }

  public train = async (images: IImage[], params: IConfigurationParams = {}) => {
    this.params = {
      ...this.params,
      ...params,
    };

    const activatedImages = await Promise.all(images.map(async (image: IImage) => {
      const img = await this.prepareData(image.data);
      return {
        activation: img,
        label: image.label,
      };
    }));

    this.data = prepareTrainingData(activatedImages);

    this.model = await train({
      ...this.data,
      classes: Object.keys(this.data.classes).length,
    }, this.params);

    return this;
  }

  public predict = async (data: tf.Tensor3D) => {
    await this.loaded();
    console.assert(this.model, 'You must call train prior to calling predict');
    const img = await this.prepareData(data);
    // TODO: Do these images need to be activated?
    const predictedClass = tf.tidy(() => {
      const predictions = this.model.predict(img);
      // TODO: address this
      return (predictions as tf.Tensor).as1D().argMax();
    });

    const classId = (await predictedClass.data())[0];
    predictedClass.dispose();
    return Object.entries(this.data.classes).reduce((obj, [
      key,
      val,
    ]) => ({
      ...obj,
      [val]: key,
    }), {})[classId];
  }

  // handlerOrURL?: tf.io.IOHandler | string;
  public save = async(handlerOrURL: string = getDefaultDownloadHandler(this.data)) => {
    await this.loaded();
    console.assert(this.model, 'You must call train prior to calling save');

    await this.model.save(handlerOrURL);

    return this;
  }
}
export default MLClassifier;
