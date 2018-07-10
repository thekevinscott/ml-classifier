import haikunator from 'haikunator';
import getDefaultDownloadHandler, {
  getName,
} from './getDefaultDownloadHandler';
jest.mock('haikunator', () => {
  return function() {
    return {
      haikunate: () => 'haikunate',
    };
  };
});

describe('getDefaultDownloadHandler', () => {
  test('it returns a url with downloads as the first argument', () => {
    expect(getDefaultDownloadHandler({}).indexOf('downloads://')).toEqual(0);
  });

  test('it returns a name for a single class', () => {
    expect(getName({
      foo: 0,
    })).toEqual('foo');
  });

  test('it returns a name for multiple classes up to 3', () => {
    expect(getName({
      foo: 0,
      baz: 2,
      bar: 1,
    })).toEqual('foo-bar-baz');
  });

  test('it returns a random name for classes greater than 3', () => {
    expect(getName({
      foo: 0,
      baz: 2,
      bar: 1,
      quom: 3,
    })).toEqual('haikunate');
  });
});
