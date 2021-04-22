import * as tf from '@tensorflow/tfjs';

class Backend{
    constructor(){

    }
    generateInput(params:any){
        //TODO get next layer params and copy if "same" convolution
        let inputSize = params.batchInputShape;
        
        if(!inputSize || !Array.isArray(inputSize)){
            throw new Error('input size not found');
        }
        let shape = inputSize.filter(v=>v).map(v=>parseInt(v))
        let out = new Float32Array((shape[0]+2)*(shape[1]+2)*(shape[2]));

        return {
            apply:(data:any)=>{
                for(let y =0;y!==shape[0];y++){
                    out.set(data.subarray(y*shape[1]*shape[2],(y+1)*shape[1]*shape[2]),(y+1)*(shape[1]+2)*shape[2]+shape[2] )
                }
                return out;
            },
            shape,
            out
        }
    }
    generateConv2d(weight:tf.Tensor<tf.Rank>[], params:any, predLayer: any){
        
        let weights = weight[0].dataSync();
        let shape:number[] = predLayer.shape.slice(0);
        shape[shape.length-1]=weight[0].shape[weight[0].shape.length-1]
        
        let outLength = 1;
        shape.forEach(v=>outLength*=v);
        
        const inpChannels = weight[0].shape[0]
        const outChannels = shape[2]
        let out = new Float32Array(outLength)
        const kernelSize = params.kernelSize;
        
        return {
            apply:new Function('data',`
            let weights = this.weights;
            for(let y =0;y<${shape[0]};y++){
                for(let x =0;x<${shape[1]};x++){
                    for(let c =0;c<${outChannels};c++){
                        let val = 0;
                                ${(()=>{
                                    let lines = [];
                        for(let dy =0;dy<kernelSize[0];dy++){
                            for(let dx =0;dx<kernelSize[1];dx++){
                               
                                    for(let z =0;z<inpChannels;z++){
                                        
                                        lines.push(`val+=data[((y+${dy})*(${shape[1]}+2) + (x+${dx}))*${inpChannels}+${z}] * weights[(${dy}*${kernelSize[0]}*${inpChannels} +${dx}*${inpChannels}+${z})*${outChannels}+c ]`);
                                    }
                               
                                
                            }
                        } 
                        return lines.join('\n')
                    })()}
                        this.out[(y*${shape[1]} + x)*${outChannels}+c] = val;
                    }
                }
            }
            return this.out;
            `).bind({weights,out}),
            // apply:(data:any)=>{
            //     for(let y =0;y<shape[0];y++){
            //         for(let x =0;x<shape[1];x++){
            //             for(let c =0;c<outChannels;c++){
            //                 let val = 0;
            //                 for(let dy =0;dy<kernelSize[0];dy++){
            //                     for(let dx =0;dx<kernelSize[1];dx++){
                                    
            //                         for(let z =0;z<inpChannels;z++){
                                        
            //                             val+= data[((y+dy)*(shape[1]+2) + (x+dx))*inpChannels+z] * weights[(dy*kernelSize[0]*inpChannels +dx*inpChannels+z)*outChannels+c ];
            //                         }
            //                     }
            //                 }
            //                 out[(y*shape[1] + x)*outChannels+c] = val;
            //             }
            //         }
            //     }
            //     return out;
            // },
            out,
            shape,
        }
    }
}

export default Backend;