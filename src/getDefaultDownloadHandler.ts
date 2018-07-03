const Haikunator = require('haikunator');

const haikunator = new Haikunator();

const getDefaultDownloadHandler = () => haikunator.haikunate();

export default getDefaultDownloadHandler;
