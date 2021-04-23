import * as tf from '@tensorflow/tfjs';
import { TypedArray } from '@tensorflow/tfjs';
import BackendCPU from './backendCPU'
//import * as tfl from '@tensorflow/tfjs-layers/src/layers/convolutional';

interface Layer {
    apply:Function,
    shape:number[],
    out:TypedArray,
    args:any[]
}
class Interpreter {
    backend='cpu';
    origModel:tf.Sequential;
    layers: any[];
    constructor(model: tf.Sequential, params:Object){
        let backend = new BackendCPU();
        this.origModel = model;
        this.layers = this.compile(model, backend);
    }
    compile=(model: tf.Sequential, backend: BackendCPU)=>{
        let layers:Layer[] = [];
        model.layers.forEach((layer, i)=>{
            
            let className = layer.getClassName();
            //if(i===4){
                console.log('className',className, i);
            //}
            if(i>21&&i<26){
            }
            switch(className){
                case('InputLayer'):{
                    layers[i] = backend.InputLayer(layer.getConfig(),  model.layers[i+1])
                    break;
                }
                case('Conv2D'):{
                    let weight = layer.getWeights();
                    let config = layer.getConfig();
                    layers[i] = backend.Conv2d(weight, config, layers[i-1], layer);
                    break;
                }
                case('MaxPooling2D'):{
                    let weight = layer.getWeights();
                    let config = layer.getConfig();

                    layers[i] = backend.MaxPooling2D(weight, config, layers[i-1], model.layers[i+1]);
                    break;
                }
                case('LeakyReLU'):{
                    let weight = layer.getWeights();
                    let config = layer.getConfig();

                    layers[i] = backend.LeakyReLU(weight, config, layers[i-1],  model.layers[i+1],layer);
                    break;
                }
                case('BatchNormalization'):{
                    let weight = layer.getWeights();
                    let config = layer.getConfig();
                    
                    layers[i] = backend.BatchNormalization(weight, config, layers[i-1]);
                    break;
                }
                default:{
                   
                    throw new Error('layer by '+className+' not supported')
                    break;
                }
            }
        })
        return layers;
    }
    
    predict = (tensor: tf.Tensor ) =>{
        //let res = this.origModel.predict(tensor)
        tensor=tensor.toFloat();

        let time = new Date().valueOf()
        let reso = (this.origModel.layers[1].apply(tensor) as tf.Tensor);
        for(let i=2;i!==22;i++){
            reso = (this.origModel.layers[i].apply(reso) as tf.Tensor);
        }
        time=new Date().valueOf()-time


        let res1=reso.dataSync();
        let tensorData = tensor.dataSync();
        let time2 = new Date().valueOf()
        let inp =this.layers[0].apply(tensorData)
        let res2 = this.layers[1].apply(inp,...this.layers[1].args);
        for(let i=2;i!==22;i++){
            res2 = this.layers[i].apply(res2,...this.layers[i].args);
        }
        console.log('time', time, new Date().valueOf()-time2);

        if(res1.length!==res2.length){
            throw new Error(`size error ${res1.length}!==${res2.length}`)

        }
        res1.forEach((v:number,i:number)=>{
            let len = Math.abs(v).toFixed(0).length;
            
            if(Math.abs(v-res2[i])>(0.001*(10**len))){
                console.log(res1[i],res2[i], i);
                throw new Error('model not equal')
            }
        })

    }


}

export default Interpreter;