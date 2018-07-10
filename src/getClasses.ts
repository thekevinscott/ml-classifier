const getClasses = (classes: string[]) => classes.reduce((labels, label) => {
  if (labels[label] !== undefined) {
    return labels;
  }

  return {
    ...labels,
    [label]: Object.keys(labels).length,
  };
}, {});

export default getClasses;
