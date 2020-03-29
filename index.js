const onnx = require('onnxjs-node');
const sharp = require('sharp');
const axios = require('axios');
const ndarray = require('ndarray');
const ops = require('ndarray-ops');
const fs = require('fs');

let hrstart = process.hrtime()

async function main(url){
  // image preprocessing
  const input = (await axios({ url: url, responseType: "arraybuffer" })).data;
  const { data, info } = await sharp(input)
    .ensureAlpha()
    .resize(224, 224)
    .raw()
    .toBuffer({ resolveWithObject: true });

  const imageData = preprocess(data,224,224)
  const result = await predict(imageData,224,224);

  return result;
}

function argMax(array) {
  return Array.from(array).map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}


function preprocess(data, width, height) {
  const dataFromImage = ndarray(new Float32Array(data), [width, height, 4]);
  // Normalize 0-255 to [0, 1]
  ops.divseq(dataFromImage, 255);
  ops.subseq(dataFromImage.pick(0, null, null), 0.485);
  ops.divseq(dataFromImage.pick(0, null, null), 0.229);
  ops.subseq(dataFromImage.pick(1, null, null), 0.456);
  ops.divseq(dataFromImage.pick(1, null, null), 0.224);
  ops.subseq(dataFromImage.pick(2, null, null), 0.406);
  ops.divseq(dataFromImage.pick(2, null, null), 0.225);

  // Realign imageData from [224*224*4] to the correct dimension [1*3*224*224].
  const dataProcessed = ndarray(new Float32Array(width * height * 3), [1, 3, height, width]);
  ops.assign(dataProcessed.pick(0, 0, null, null), dataFromImage.pick(null, null, 0));
  ops.assign(dataProcessed.pick(0, 1, null, null), dataFromImage.pick(null, null, 1));
  ops.assign(dataProcessed.pick(0, 2, null, null), dataFromImage.pick(null, null, 2));

  return dataProcessed.data;
}


async function predict(preprocessedData, modelWidth=224, modelHeight=224){
  let hrend = process.hrtime(hrstart)
  const modelName = 'pdxTrees';
  const model = `./models/${modelName}.onnx`;
  const classLabels = JSON.parse(fs.readFileSync(`./labels/${modelName}.json`,'utf-8'))["classLabels"];

  const session = new onnx.InferenceSession();
  await session.loadModel(model);
  const inputTensor = new onnx.Tensor(preprocessedData, 'float32', [1, 3, modelWidth, modelHeight]);
  const classProbabilities = await session.run([inputTensor]);
  const outputData = classProbabilities.values().next().value.data;
  console.log('Execution time (hr): %ds %dms', hrend[0], hrend[1] / 1000000)
  console.log(hrend[1],' nano seconds')
  return classLabels[argMax(outputData)];
}

main('https://i.pinimg.com/originals/fb/13/dd/fb13dd2ab6cc760b3ac5ca15b9de11a6.jpg').then((result)=>{
  console.log(result)
});

