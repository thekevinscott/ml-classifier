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

interface IArgs {
  onLoadStart?: Function;
  onAddDataStart?: Function;
  onClearDataStart?: Function;
  onTrainStart?: Function;
  onEvaluateStart?: Function;
  onSaveStart?: Function;
  onPredictStart?: Function;
  onLoadComplete?: Function;
  onAddDataComplete?: Function;
  onClearDataComplete?: Function;
  onTrainComplete?: Function;
  onEvaluateComplete?: Function;
  onSaveComplete?: Function;
  onPredictComplete?: Function;
}

// export { DataType } from './types';

class MLClassifier {
  // private pretrainedModel: typeof tf.model;
  private pretrainedModel: any;
  // private model: tf.Sequential;
  private model: any;
  private callbacks: Function[] = [];
  private args: IArgs;
  private data: IData = {
    classes: {},
  };

  constructor(args: IArgs) {
    this.args = args;
    this.init();
  }

  private callbackFn = (fn: string, type: string, ...args: any[]) => {
    const key = `${fn}${type.substring(0, 1).toUpperCase()}${type.substring(1)}`;
    if (this.args[key] && typeof this.args[key] === 'function') {
      this.args[key](...args);
    }
  }

  private init = async () => {
    this.callbackFn('onLoad', 'start');
    this.pretrainedModel = await loadPretrainedModel(PRETRAINED_MODELS_KEYS.MOBILENET);

    this.callbacks.map(callback => callback());

    this.callbackFn('onLoad', 'complete');
  }

  private loaded = async () => new Promise(resolve => {
    if (this.pretrainedModel) {
      return resolve();
    }

    this.callbacks.push(() => {
      resolve();
    });
  });

  // private cropAndActivateImage = async (image: tf.Tensor3D) => {
  private cropAndActivateImage = async (image: any) => {
    await this.loaded();
    const {
      inputLayers,
    } = this.pretrainedModel;
    const {
      batchInputShape,
    } = inputLayers[0];
    const dims = await this.getInputDims();
    const processedImage = await cropAndResizeImage(image, dims);
    return this.pretrainedModel.predict(processedImage);
  }

  private getInputDims = async (): Promise<[number, number]> => {
    await this.loaded();

    const {
      inputLayers,
    } = this.pretrainedModel;

    const {
      batchInputShape,
    } = inputLayers[0];

    return [
      batchInputShape[1],
      batchInputShape[2],
    ];
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

  public addData = async (origImages: Array<tf.Tensor3D | HTMLImageElement>, origLabels: string[], dataType: string = 'train') => {
    this.callbackFn('onAddData', 'start', origImages, origLabels, dataType);
    if (!origImages) {
      throw new Error('You must supply images');
    }
    if (!origLabels) {
      throw new Error('You must supply labels');
    }

    const dims = await this.getInputDims();
    const {
      images,
      errors,
      labels,
    } = await translateImages(origImages, dims, origLabels);
    if (images.length !== labels.length) {
      throw new Error('Class mismatch between labels and images');
    }

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

    this.callbackFn('onAddData', 'complete', origImages, labels, dataType, errors);
  }

  public clearData = async (dataType?: string) => {
    this.callbackFn('onClearData', 'start', dataType);
    if (dataType) {
      this.data[dataType] = { };
    }

    this.data['train'] = {};
    this.data['eval'] = {};

    this.callbackFn('onClearData', 'complete', dataType);
  }

  public train = async (params: IParams = {}) => {
    this.callbackFn('onTrain', 'start', params);
    await this.loaded();
    const data = await this.getData('train');

    if (!data.xs) {
      throw new Error('You must add some training examples');
    }
    if (!data.ys) {
      throw new Error('You must add some training labels');
    }

    const classes = Object.keys(data.classes).length;
    if (classes <= 1) {
      throw new Error('You must train with more than one class');
    }
    const {
      model,
      history,
    } = await train(data, classes, params);

    this.model = model;
    this.callbackFn('onTrain', 'complete', params, history);
    return history;
  }

  public predict = async (origImage: tf.Tensor3D | HTMLImageElement | string, label?: string) => {
    this.callbackFn('onPredict', 'start', origImage);
    await this.loaded();
    if (!this.model) {
      throw new Error('You must call train prior to calling predict');
    }
    const dims = await this.getInputDims();
    const {
      images,
      errors,
    } = await translateImages([origImage], dims);
    if (errors && errors.length) {
      throw errors[0].error;
    }
    const data = images[0];
    const img = await this.cropAndActivateImage(data);
    // TODO: Do these images need to be activated?
    const predictedClass = tf.tidy(() => {
      const predictions = this.model.predict(img);
      // TODO: address this
      return (predictions as tf.Tensor).as1D().argMax();
    });

    const classId = (await predictedClass.data())[0];
    predictedClass.dispose();
    const prediction = Object.entries(this.data.classes).reduce((obj, [
      key,
      val,
    ]) => ({
      ...obj,
      [val]: key,
    }), {})[classId];
    this.callbackFn('onPredict', 'complete', origImage, label, prediction);
    return prediction;
  }

  public evaluate = async (params: IParams = {}) => {
    this.callbackFn('onEvaluate', 'start', params);
    await this.loaded();
    if (!this.model) {
      throw new Error('You must call train prior to calling predict');
    }
    const data = await this.getData('eval');

    if (!data.xs || !data.ys) {
      throw new Error('You must add some evaluation examples');
    }

    const evaluation = await this.model.evaluate(data.xs, data.ys, params);
    this.callbackFn('onEvaluate', 'complete', params, evaluation);
    return evaluation;
  }

  // handlerOrURL?: tf.io.IOHandler | string;
  public save = async(handlerOrURL?: string, params: IParams = {}) => {
    this.callbackFn('onSave', 'start', handlerOrURL, params);
    await this.loaded();
    if (!this.model) {
      throw new Error('You must call train prior to calling save');
    }

    const savedModel = await this.model.save(handlerOrURL || getDefaultDownloadHandler(this.data.classes), params);
    this.callbackFn('onSave', 'complete', handlerOrURL, params, savedModel);
    return savedModel;
  }
}

export default MLClassifier;
