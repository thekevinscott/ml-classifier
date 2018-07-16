import * as tf from '@tensorflow/tfjs';
import cropAndResizeImage from './cropAndResizeImage';
import getClasses from './getClasses';
import train from './train';
import translateImages from './translateImages';
import loadPretrainedModel, {
  PRETRAINED_MODELS_KEYS,
} from './loadPretrainedModel';
import {
  addData,
  addLabels,
} from './prepareTrainingData';
import getDefaultDownloadHandler from './getDefaultDownloadHandler';

import {
  IParams,
  // IConfigurationParams,
  IData,
  ICollectedData,
  // DataType,
} from './types';

// export { DataType } from './types';

class MLClassifier {
  // private pretrainedModel: typeof tf.model;
  private pretrainedModel: any;
  // private model: tf.Sequential;
  private model: any;
  private callbacks: Function[] = [];
  private data: IData = {
    classes: {},
  };

  constructor() {
    this.init();
  }

  private init = async () => {
    this.pretrainedModel = await loadPretrainedModel(PRETRAINED_MODELS_KEYS.MOBILENET);

    this.callbacks.map(callback => callback());
  }

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
    const processedImage = await cropAndResizeImage(image);
    return this.pretrainedModel.predict(processedImage);
  }

  private getData = async (dataType: string): Promise<ICollectedData> => {
    if (dataType !== 'train' && dataType !== 'eval') {
      throw new Error(`Datatype ${dataType} unsupported`);
    }

    return {
      xs: this.data[dataType].xs,
      ys: this.data[dataType].ys,
      classes: this.data.classes,
    };
  }

  public getModel = () => this.model;

  public addData = async (origImages: Array<tf.Tensor3D | HTMLImageElement>, labels: string[], dataType: string = 'train') => {
    if (!origImages) {
      throw new Error('You must supply images');
    }
    if (!labels) {
      throw new Error('You must supply labels');
    }

    const images = await translateImages(origImages);

    if (dataType === 'train' || dataType === 'eval') {
      const activatedImages = await Promise.all(images.map(async (image: tf.Tensor3D, idx: number) => {
        await tf.nextFrame();
        return await this.cropAndActivateImage(image);
      }));

      this.data.classes = getClasses(labels);
      const xs = addData(activatedImages);
      const ys = addLabels(labels, this.data.classes);
      this.data[dataType] = {
        xs,
        ys,
      };
    }
  }

  public clearData = async (dataType?: string) => {
    if (dataType) {
      this.data[dataType] = { };
    }

    this.data['train'] = {};
    this.data['eval'] = {};
  }

  public train = async (params: IParams = {}) => {
    await this.loaded();
    const data = await this.getData('train');

    if (!data.xs) {
      throw new Error('You must add some training examples');
    }

    const classes = Object.keys(data.classes).length;
    const {
      model,
      history,
    } = await train(data, classes, params);

    this.model = model;
    return history;
  }

  public predict = async (data: tf.Tensor3D) => {
    await this.loaded();
    if (!this.model) {
      throw new Error('You must call train prior to calling predict');
    }
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

  public evaluate = async (params: IParams = {}) => {
    await this.loaded();
    if (!this.model) {
      throw new Error('You must call train prior to calling predict');
    }
    const data = await this.getData('eval');

    if (!data.xs || !data.ys) {
      throw new Error('You must add some evaluation examples');
    }

    return await this.model.evaluate(data.xs, data.ys, params);
  }

  // handlerOrURL?: tf.io.IOHandler | string;
  public save = async(handlerOrURL?: string, params: IParams = {}) => {
    await this.loaded();
    if (!this.model) {
      throw new Error('You must call train prior to calling save');
    }

    return await this.model.save(handlerOrURL || getDefaultDownloadHandler(this.data.classes), params);
  }
}

export default MLClassifier;
