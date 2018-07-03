const Haikunator = require('haikunator');

const haikunator = new Haikunator();

const getDefaultDownloadHandler = () => `downloads://${haikunator.haikunate()}`;

export default getDefaultDownloadHandler;
