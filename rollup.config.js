// rollup.config.js
import typescript from 'rollup-plugin-typescript2';

export default {
  entry: './src/index.ts',
  output: {
    name: 'MLClassifier',
    file: './dist/index.js',
    format: 'umd',
  },
  plugins: [
    typescript({
    }),
  ],
};

