import getClasses from './getClasses';

describe('getClasses', () => {
  it('returns a single label for a single image', () => {
    expect(getClasses([{
      label: 'foo',
      data: null,
    }])).toEqual({ foo: 0 });
  });

  it('returns multiple labels for multiple images in the order they are recieved', () => {
    expect(getClasses([{
      label: 'foo',
      data: null,
    }, {
      label: 'bar',
      data: null,
    }, {
      label: 'baz',
      data: null,
    }])).toEqual({ foo: 0, bar: 1, baz: 2 });
  });

  it('returns a list of unique labels for a set of images', () => {
    expect(getClasses([{
      label: 'foo',
      data: null,
    }, {
      label: 'bar',
      data: null,
    }, {
      label: 'foo',
      data: null,
    }, {
      label: 'foo',
      data: null,
    }, {
      label: 'bar',
      data: null,
    }])).toEqual({ foo: 0, bar: 1 });
  });
});
