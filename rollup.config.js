// rollup.config.js
import typescript from 'rollup-plugin-typescript2';

export default {
  input: './src/index.ts',
  output: {
    name: 'MLClassifier',
    file: './dist/index.js',
    format: 'es',
    external: [
      '@tensorflow/tfjs',
    ],
  },
  plugins: [
    typescript({
      tsconfig: 'tsconfig.json',
      verbosity: 1,
      exclude: [
        '*.d.ts',
        '**/*.d.ts',
      ],
    }),
  ],
};

