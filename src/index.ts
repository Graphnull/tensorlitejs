import * as tf from '@tensorflow/tfjs';
import { TypedArray } from '@tensorflow/tfjs';
import BackendCPU from './backendCPU'
//import * as tfl from '@tensorflow/tfjs-layers/src/layers/convolutional';

interface Layer {
    apply:Function,
    shape:number[],
    out:TypedArray,
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
            if(i>1){
                return;
            }
            switch(layer.getClassName()){
                case('InputLayer'):{
                    let func = backend.generateInput(layer.getConfig())
                    layers[i] = func
                    break;
                }
                case('Conv2D'):{
                    let weight = layer.getWeights();
                    let config = layer.getConfig();
                    //console.log(weight[0].dataSync());
                    
                    let func = backend.generateConv2d(weight, config, layers[i-1]);
                    
                    layers[i] = func
                    //process.exit()
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
        time=new Date().valueOf()-time
        let res1=reso.dataSync();
        let tensorData = tensor.dataSync();
        let time2 = new Date().valueOf()
        let inp =this.layers[0].apply(tensorData)
        let res2 = this.layers[1].apply(inp);

        console.log('time', time, new Date().valueOf()-time2);
        res1.forEach((v:number,i:number)=>{
            if(Math.abs(v-res2[i])>0.001){
                console.log(v,res2[i], i);
                throw new Error('model not equal')
                process.exit(1)
            }
        })

    }


}

export default Interpreter;