import * as tf from '@tensorflow/tfjs';
import prepareData from './prepareData';
import train from './train';
import loadPretrainedModel, {
  PRETRAINED_MODELS_KEYS,
} from './loadPretrainedModel';

interface IParams {
  epochs?: number;
  LR?: number;
  optimizer?: string;
  onTrainBegin?: Function;
  onTrainEnd?: Function;
  onEpochBegin?: Function;
  onEpochEnd?: Function;
  onBatchBegin?: Function;
  onBatchEnd?: Function;
}

const defaultParams = {
  epochs: 20,
  LR: 0.0001,
  optimizer: 'adam',
};


let xs: tf.Tensor3D | null = null;
let ys: tf.Tensor2D | null = null;

export const addData = async(example: tf.Tensor3D, label: number, classes: number) => {
  // One-hot encode the label.
  const y = tf.tidy(() => tf.oneHot(tf.tensor1d([label]).toInt(), classes));

  if (xs === null || ys === null) {
    // For the first example that gets added, keep example and y so that the
    // ControllerDataset owns the memory of the inputs. This makes sure that
    // if addExample() is called in a tf.tidy(), these Tensors will not get
    // disposed.
    xs = tf.keep(example);
    ys = tf.keep(y);
  } else {
    const oldX = xs;
    xs = tf.keep(oldX.concat(example, 0));

    const oldY = ys;
    ys = tf.keep(oldY.concat(y, 0));

    oldX.dispose();
    oldY.dispose();
    y.dispose();
  }
}

interface IImage {
  label: string;
  data: tf.Tensor3D;
}

class MLClassifier {
  private params: IParams;
  // private pretrainedModel: typeof tf.model;
  // private model: typeof tf.model;
  private pretrainedModel: any;
  private model: tf.Sequential;
  private callbacks: Function[] = [];
  private data: {
    classes: string[];
  };
  public tf = tf;

  constructor(params: IParams = {}) {
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

  public train = async (images: IImage[]) => {
    const startingArray: string[] = [];
    this.data = {
      classes: images.reduce((arr, { label }) => {
        if (arr.includes(label)) {
          return arr;
        }

        return arr.concat(label);
      }, startingArray),
    };

    console.log(this.data.classes);

    const loadImage = async (image: IImage) => {
      const img = await this.prepareData(image.data);
      return tf.tidy(() => {
        addData(img, this.data.classes[image.label], this.data.classes.length);
      });
    };

    await Promise.all(images.map((image: IImage) => {
      // console.log('data', data, label);
      return loadImage(image);
    }));

    this.model = await train({
      xs,
      ys,
      classes: this.data.classes.length,
    }, [
      'onTrainBegin',
      'onTrainEnd',
      'onEpochBegin',
      'onEpochEnd',
      'onBatchBegin',
      'onBatchEnd',
    ].reduce((callbacks, key) => {
      if (this.params[key]) {
        return {
          ...callbacks,
          [key]: this.params[key],
        };
      }

      return callbacks;
    }, {}));

    return this;
  }

  public predict = async (data: tf.Tensor3D) => {
    await this.loaded();
    const img = await this.prepareData(data);
    const predictedClass = tf.tidy(() => {
      const predictions = this.model.predict(img);
      console.log('fix this');
      return (predictions as tf.Tensor).as1D().argMax();
    });

    const classId = (await predictedClass.data())[0];
    predictedClass.dispose();
    return classId;
    // return this.data.classes[classId];
  }
}

export default MLClassifier;
