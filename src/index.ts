import * as tf from '@tensorflow/tfjs';
import prepareData from './prepareData';
import getClasses from './getClasses';
import train from './train';
import loadPretrainedModel, {
  PRETRAINED_MODELS_KEYS,
} from './loadPretrainedModel';
import prepareTrainingData from './prepareTrainingData';
import getDefaultDownloadHandler from './getDefaultDownloadHandler';

console.log('i am the ml classifier v2');

import {
  IImage,
  IConfigurationParams,
  IParams,
  IData,
  ICollectedData,
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
  private data: IData;
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
  public getParams = () => this.params;

  private loaded = async () => new Promise(resolve => {
    if (this.pretrainedModel) {
      return resolve();
    }

    this.callbacks.push(() => {
      resolve();
    });
  });

  private cropAndActivateImage = async (image: tf.Tensor3D) => {
    await this.loaded();
    const processedImage = await prepareData(image);
    return this.pretrainedModel.predict(processedImage);
  }

  public addData = async (images: IImage[]) => {
    const activatedImages = await Promise.all(images.map(async (image: IImage) => {
      const img = await this.cropAndActivateImage(image.data);
      return {
        activation: img,
        label: image.label,
      };
    }));

    this.data.classes = getClasses(images);
    this.data.train = prepareTrainingData(activatedImages, this.data.classes);
  }

  private getData = async (dataType: string): Promise<ICollectedData> => {
    if (!this.data[dataType]) {
      throw new Error(`Datatype ${dataType} unsupported`);
    }

    return {
      xs: this.data[dataType].xs,
      ys: this.data[dataType].ys,
      classes: this.data.classes,
    };
  }

  public train = async (params: IConfigurationParams = {}) => {
    const data = await this.getData('train');

    if (!data.xs) {
      throw new Error('You must add some training examples');
    }

    const classes = Object.keys(data.classes).length;
    try {
      this.model = await train(data, classes, {
        ...this.params,
        ...params,
      });
    } catch(err) {
      throw err;
    }
  }

  public predict = async (data: tf.Tensor3D) => {
    await this.loaded();
    console.assert(this.model, 'You must call train prior to calling predict');
    const img = await this.cropAndActivateImage(data);
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

  // public evaluate = async (images: IImage[], params: IConfigurationParams = {}) => {
  //   await this.loaded();
  //   console.assert(this.model, 'You must call train prior to calling evaluate');
  //   const combinedParams = {
  //     ...this.params,
  //     ...params,
  //   };

  //   const activatedImages = await Promise.all(images.map(async (image: IImage) => {
  //     const img = await this.cropAndActivateImage(image.data);
  //     return {
  //       activation: img,
  //       label: image.label,
  //     };
  //   }));

  //   const {
  //     xs,
  //     ys,
  //   } = prepareTrainingData(activatedImages);

  //   if (xs !== undefined && ys !== undefined) {
  //     console.assert(this.model, 'You must call train prior to calling save');
  //     return await this.model.evaluate({
  //       xs,
  //       ys,
  //     }, {
  //       batchSize: combinedParams.batchSize,
  //       verbose: combinedParams.verbose,
  //       sampleWeight: combinedParams.sampleWeight,
  //       steps: combinedParams.steps,
  //     });
  //   }

  //   return this;
  // }

  // handlerOrURL?: tf.io.IOHandler | string;
  public save = async(handlerOrURL?: string) => {
    await this.loaded();
    if (!this.model) {
      console.log('hello!');
      throw new Error('You must call train prior to calling save');
    }

    return await this.model.save(handlerOrURL || getDefaultDownloadHandler(this.data.classes));
  }
}
export default MLClassifier;
