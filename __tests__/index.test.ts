import MLClassifier from '../dist/index';

describe('Integration test', () => {
  test('that it initializes', () => {
    const mlClassifier = new MLClassifier();
    expect(mlClassifier).toBeDefined();
  });

  test('that it demands images', () => {
    const mlClassifier = new MLClassifier();
    return mlClassifier.addData().catch(err => {
      expect(err.message).toEqual('You must supply images');
    });
  });

  test('that it demands labels', () => {
    const mlClassifier = new MLClassifier();
    return mlClassifier.addData([]).catch(err => {
      expect(err.message).toEqual('You must supply labels');
    });
  });
});
