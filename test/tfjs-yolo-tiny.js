let  tf =require('@tensorflow/tfjs');
let fs = require('fs')
let downloadModel = 'https://raw.githubusercontent.com/MikeShi42/yolo-tiny-tfjs/master/model2.json';

let Interpreter = require('./../src/index').default;

let load=async ()=>{
    let model = await tf.loadLayersModel(downloadModel);
    let result = await model.save(tf.io.withSaveHandler(async modelArtifacts => modelArtifacts));
    result.weightData = Buffer.from(result.weightData).toString("base64");
const jsonStr = JSON.stringify(result);
fs.writeFileSync('./test/yoloModel.bin', jsonStr)
return model
}
let test = async ()=>{

    let img =  fs.readFileSync('./test/416x416.ppm');
    let tensor = tf.tensor4d(img.slice(img.length-416*416*3), [1,416,416,3])
    
    let model;
    try{
    const json = JSON.parse(fs.readFileSync('./test/yoloModel.bin').toString());
    const weightData = new Uint8Array(Buffer.from(json.weightData, "base64")).buffer;
    model = await tf.loadLayersModel(tf.io.fromMemory(json.modelTopology, json.weightSpecs, weightData));
    }catch(err){
        model = await load()
    }

    
    let lite = new Interpreter(model)
    console.log(lite.predict(tensor));
    //let result = model.predict(tensor)
}

test();
