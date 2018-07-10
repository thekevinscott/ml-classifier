import * as tf from '@tensorflow/tfjs';
import cropAndResizeImage from './cropAndResizeImage';
import getClasses from './getClasses';
import train from './train';
import loadPretrainedModel, {
  PRETRAINED_MODELS_KEYS,
} from './loadPretrainedModel';
import {
  addData,
  addLabels,
} from './prepareTrainingData';
import getDefaultDownloadHandler from './getDefaultDownloadHandler';

console.log('i am the ml classifier v2');

import {
  IConfigurationParams,
  IParams,
  IData,
  ICollectedData,
  DataType,
} from './types';

const defaultParams = {
  epochs: 20,
  loss: 'categoricalCrossentropy',
  optimizer: tf.train.adam(0.0001),
  callbacks: {},
};

export { DataType } from './types';;

class MLClassifier {
  private params: IParams;
  // private pretrainedModel: typeof tf.model;
  private pretrainedModel: any;
  private model: tf.Sequential;
  private callbacks: Function[] = [];
  private data: IData = {
    classes: {},
  };
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
    const processedImage = await cropAndResizeImage(image);
    return this.pretrainedModel.predict(processedImage);
  }

  public addData = async (images: tf.Tensor3D[], labels?: string[], dataType?: DataType) => {
    if (!dataType) {
      if (typeof labels === 'string') {
        dataType = labels;
      } else {
        dataType = DataType.TRAIN;
      }
    }

    if (dataType === DataType.TRAIN || dataType === DataType.EVAL) {
      if (!labels) {
        throw new Error(`You must provide labels when supplying ${dataType} data`);
      }

      const activatedImages = await Promise.all(images.map(async (image: tf.Tensor3D, idx: number) => {
        return await this.cropAndActivateImage(image);
      }));

      this.data.classes = getClasses(labels);
      const xs = addData(activatedImages);
      const ys = addLabels(labels, this.data.classes);
      this.data[dataType] = {
        xs,
        ys,
      };
    } else if (dataType === DataType.EVAL) {
      const activatedImages = await Promise.all(images.map(async (image: tf.Tensor3D, idx: number) => {
        return await this.cropAndActivateImage(image);
      }));

      const xs = addData(activatedImages);

      this.data[dataType] = {
        xs,
      };
    }
  }

  public clearData = async (dataType?: DataType) => {
    if (dataType) {
      this.data[dataType] = { };
    }

    this.data[DataType.TRAIN] = {};
    this.data[DataType.EVAL] = {};
  }

  private getData = async (dataType: DataType): Promise<ICollectedData> => {
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
    await this.loaded();
    const data = await this.getData(DataType.TRAIN);

    if (!data.xs) {
      throw new Error('You must add some training examples');
    }

    const classes = Object.keys(data.classes).length;
    const {
      model,
      history,
    } = await train(data, classes, {
      ...this.params,
      ...params,
    });

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

  public evaluate = async (params: IConfigurationParams = {}) => {
    await this.loaded();
    if (!this.model) {
      throw new Error('You must call train prior to calling predict');
    }
    const data = await this.getData(DataType.EVAL);

    if (!data.xs || !data.ys) {
      throw new Error('You must add some evaluation examples');
    }

    const combinedParams = {
        ...this.params,
        ...params,
    };

    return await this.model.evaluate(data.xs, data.ys, {
      batchSize: combinedParams.batchSize,
      verbose: combinedParams.verbose,
      sampleWeight: combinedParams.sampleWeight,
      steps: combinedParams.steps,
    });
  }

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
