const onnx = require('onnxjs-node');
const sharp = require('sharp');
const axios = require('axios');
const ndarray = require('ndarray');
const ops = require('ndarray-ops');

let hrstart = process.hrtime()

async function main(url){
  let hrend = process.hrtime(hrstart)
  // image preprocessing
  const input = (await axios({ url: url, responseType: "arraybuffer" })).data;
  const { data, info } = await sharp(input)
    .resize(224, 224)
    .raw()
    .toBuffer({ resolveWithObject: true });
  // const incomingData = new Uint8Array(data);
  // const float32 = new Float32Array(data);
  // const float32 = Float32Array.from(incomingData);

  const dataFromImage = ndarray(new Float32Array(data), [info.width, info.height, 4]);
  ops.divseq(dataFromImage, 255);
  ops.subseq(dataFromImage.pick(0, null, null), 0.485);
  ops.divseq(dataFromImage.pick(0, null, null), 0.229);
  ops.subseq(dataFromImage.pick(1, null, null), 0.456);
  ops.divseq(dataFromImage.pick(1, null, null), 0.224);
  ops.subseq(dataFromImage.pick(2, null, null), 0.406);
  ops.divseq(dataFromImage.pick(2, null, null), 0.225);
  
  //  [224,224,4] => [1,3,224,224]
  const dataProcessed = ndarray(new Float32Array(info.width * info.height * 3), [1, 3, info.height, info.width]);
  // ops.assign(dataProcessed.pick(0, 0, null, null), dataFromImage.pick(null, null, 0));
  // ops.assign(dataProcessed.pick(0, 1, null, null), dataFromImage.pick(null, null, 1));
  // ops.assign(dataProcessed.pick(0, 2, null, null), dataFromImage.pick(null, null, 2));

  const tensor = new onnx.Tensor(dataProcessed.data, 'float32', [1, 3, info.width, info.height]);
  // const tensor = new onnx.Tensor(float32, 'float32', [1, 3, info.width, info.height]);
  // console.log(tensor)

  // inference
  const session = new onnx.InferenceSession();
  const model = "./bestmodel.onnx";
  await session.loadModel(model);
  const outputMap = await session.run([tensor]);
  console.log(outputMap.values())
  console.log('Execution time (hr): %ds %dms', hrend[0], hrend[1] / 1000000)
  console.log(hrend[1],' nano seconds')

}

main('https://b1.pngbarn.com/png/412/963/tree-16-palm-tree-png-clip-art-thumbnail.png')
// main('http://storage.needpix.com/rsynced_images/tree-1439369_1280.jpg')

