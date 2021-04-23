import * as tf from '@tensorflow/tfjs';
import {backend_util, util} from '@tensorflow/tfjs-core'
class Backend {
    constructor() {

    }
    InputLayer(params: any, nextLayer: any) {
        //TODO get next layer params and copy if "same" convolution
        let inputSize = params.batchInputShape;

        if (!inputSize || !Array.isArray(inputSize)) {
            throw new Error('input size not found');
        }
        let shape = inputSize.filter(v => v).map(v => parseInt(v))
        let out = new Float32Array((shape[0] + 2) * (shape[1] + 2) * (shape[2]));

        if (nextLayer && nextLayer.padding === 'same') {
            return {
                args:[],
                apply: (data: any) => {
                    for (let y = 0; y !== shape[0]; y++) {
                        out.set(data.subarray(y * shape[1] * shape[2], (y + 1) * shape[1] * shape[2]), (y + 1) * (shape[1] + 2) * shape[2] + shape[2])
                    }
                    return out;
                },
                shape,
                out
            }
        } else {
            return {
                args:[],
                apply: (data: any) => data,
                shape,
                out
            }
        }
    }
    LeakyReLU(weight: tf.Tensor<tf.Rank>[], params: any, predLayer: any, nextLayer:any,layer:any) {
        

        let shape = predLayer.shape; //inputSize.filter(v=>v).map(v=>parseInt(v))
        
        
        if (nextLayer && nextLayer.getClassName()==='Conv2D' && nextLayer.padding === 'same') {
            let out = new Float32Array((shape[0]+2) * (shape[1]+2) * (shape[2]));
            return {
                args:[],
                apply:(data:any)=>{
                    console.log('shape: ', shape,out.length, data.length);
                        for(let i =0;i<out.length;i++){
                            if(data[i]<0){
                                out[i+(shape[1]+2)*shape[2] + shape[2]] = data[i] *params.alpha
                            }else{
                                out[i+(shape[1]+2)*shape[2] + shape[2]] = data[i]
                            }
                        }
                        return out;
                    },
                shape,
                out
            }
        } else {
            let out = new Float32Array((shape[0]) * (shape[1]) * (shape[2]));
            return {
                args:[],
                apply: new Function('data', `
                for(let i =0;i<${out.length};i++){
                    if(data[i]<0){
                        data[i] = data[i] *${params.alpha}
                    }
                }
                return data;
            `),
                // apply:(data:any)=>{
                //     for(let i =0;i<out.length;i++){
                //         if(data[i]<0){
                //             data[i] = data[i] *params.alpha
                //         }
                //     }
                //     return data;
                // },
                shape,
                out
            }
        }
    }
    MaxPooling2D(weight: tf.Tensor<tf.Rank>[], params: any, predLayer: any, nextLayer: any) {

        let shape = predLayer.shape;
        let origW = shape[shape.length - 2]
        shape[shape.length - 2] = Math.floor(shape[shape.length - 2] / 2);
        shape[shape.length - 3] = Math.floor(shape[shape.length - 3] / 2);
        let ox = 0;
        let oy = 0;
        if (nextLayer && nextLayer.padding === 'same') {
            ox = 1;
            oy = 1;
        }
        let out = new Float32Array((shape[0] + oy * 2) * (shape[1] + ox * 2) * (shape[2]));

        let c = shape[shape.length - 1]
        let w = shape[shape.length - 2]
        let h = shape[shape.length - 3];

        return {
            args:[],
            apply: new Function('data', `
            for(let y =0;y<${h};y++){
                for(let x =0;x<${w};x++){
                    for(let z =0;z<${c};z++){
                        this.out[((y+${oy})*${(w + ox * 2)}+x+${ox})*${c}+z] = Math.max(
                            data[((y*2+0)*${origW}+(x*2+0))*${c}+z],
                            data[((y*2+0)*${origW}+(x*2+1))*${c}+z],
                            data[((y*2+1)*${origW}+(x*2+0))*${c}+z],
                            data[((y*2+1)*${origW}+(x*2+1))*${c}+z]
                        )
                    }
                }
            }
            return this.out;
        `).bind({ out }),
            // apply:(data:any)=>{
            //     for(let y =0;y<h;y++){
            //         for(let x =0;x<w;x++){
            //             for(let z =0;z<c;z++){
            //                 out[((y+oy)*(w+ox*2)+x+ox)*c+z] = Math.max(
            //                     data[(y*2+0)*origW*c+(x*2+0)*c+z],
            //                     data[(y*2+0)*origW*c+(x*2+1)*c+z],
            //                     data[(y*2+1)*origW*c+(x*2+0)*c+z],
            //                     data[(y*2+1)*origW*c+(x*2+1)*c+z]
            //                 )
            //             }
            //         }
            //     }
            //     return out;
            // },
            shape,
            out
        }
    }
    OrigConv2d(weight: tf.Tensor<tf.Rank>[], params: any, predLayer: any, currentLayer: any) {

        let {strides, pad, dataFormat, dilations, dimRoundingMode} = params;
        let weights = weight[0].dataSync();
        let shape: number[] = predLayer.shape.slice(0);
        shape[shape.length - 1] = weight[0].shape[weight[0].shape.length - 1]
        let filter = weight[0];
        let outLength = 1;
        shape.forEach(v => outLength *= v);


        let out = new Float32Array(outLength)
        
        pad=params.padding
        dilations=params.dilationRate
        const $dataFormat = dataFormat;
        const convInfo = backend_util.computeConv2DInfo(
            [1].concat(shape) as [number, number, number, number],
            filter.shape as [number, number, number, number], strides, dilations, pad,
            dimRoundingMode, false /* depthwise */, $dataFormat);
      
        const filterHeight = convInfo.filterHeight;
        const filterWidth = convInfo.filterWidth;
        const dilationHeight = convInfo.dilationHeight;
        const dilationWidth = convInfo.dilationWidth;
        const padLeft = convInfo.padInfo.left;
        const padTop = convInfo.padInfo.top;
        const isChannelsLast = convInfo.dataFormat === 'channelsLast';
      
        const y = new tf.TensorBuffer(convInfo.outShape, 'float32');
      
        const xStrides = util.computeStrides(shape);
        const filterStrides = util.computeStrides(filter.shape);
      
        const xBatchStride = xStrides[0];
        const xRowStride = isChannelsLast ? xStrides[1] : xStrides[2];
        const xColStride = isChannelsLast ? xStrides[2] : 1;
        const xChannelStride = isChannelsLast ? 1 : xStrides[1];
        const yBatchStride = y.strides[0];
        const yRowStride = isChannelsLast ? y.strides[1] : y.strides[2];
        const yColStride = isChannelsLast ? y.strides[2] : 1;
        const yChannelStride = isChannelsLast ? 1 : y.strides[1];

        let apply:any = (data: any) => {
                for (let b = 0; b < convInfo.batchSize; ++b) {
                    const xOffset1 = b * xBatchStride;
                    const yOffset1 = b * yBatchStride;
                    for (let yR = 0; yR < convInfo.outHeight; ++yR) {
                        const yOffset2 = yOffset1 + yR * yRowStride;
                        const xRCorner = yR * convInfo.strideHeight - padTop;
                        for (let wR = 0; wR < filterHeight; ++wR) {
                            const xR = xRCorner + wR * dilationHeight;
                            if (xR < 0 || xR >= convInfo.inHeight) {
                                continue;
                            }
                            const wOffset1 = wR * filterStrides[0];
                            const xOffset2 = xOffset1 + xR * xRowStride;
                            for (let yC = 0; yC < convInfo.outWidth; ++yC) {
                                const yOffset3 = yOffset2 + yC * yColStride;
                                const xCCorner = yC * convInfo.strideWidth - padLeft;
                                for (let wC = 0; wC < filterWidth; ++wC) {
                                    const xC = xCCorner + wC * dilationWidth;
                                    if (xC < 0 || xC >= convInfo.inWidth) {
                                        continue;
                                    }
                                    const wOffset2 = wOffset1 + wC * filterStrides[1];
                                    const xOffset3 = xOffset2 + xC * xColStride;
                                    let wOffset3 = wOffset2;
                                    for (let d1 = 0; d1 < convInfo.inChannels; ++d1) {
                                        const xVal = data[xOffset3 + d1 * xChannelStride];
                                        for (let d2 = 0; d2 < convInfo.outChannels; ++d2) {
                                            out[yOffset3 + d2 * yChannelStride] +=
                                                xVal * weights[wOffset3 + d2];
                                        }
                                        wOffset3 += convInfo.outChannels;
                                    }
                                }
                            }
                        }
                    }
                }
                return out;
            }
        let funcStr = apply.toString().split('\n').slice(1,-1).join('\n')
        .replace(/convInfo\.batchSize/gi, ''+convInfo.batchSize)
        .replace(/xBatchStride/gi, ''+xBatchStride)
        .replace(/yBatchStride/gi, ''+yBatchStride)
        .replace(/convInfo\.outHeight/gi, ''+convInfo.outHeight)
        .replace(/convInfo\.strideHeight/gi, ''+convInfo.strideHeight)
        .replace(/convInfo\.inHeight/gi, ''+convInfo.inHeight)
        .replace(/convInfo\.outWidth/gi, ''+convInfo.outWidth)
        .replace(/convInfo\.strideWidth/gi, ''+convInfo.strideWidth)
        .replace(/convInfo\.inWidth/gi, ''+convInfo.inWidth)
        .replace(/convInfo\.inChannels/gi, ''+convInfo.inChannels)
        .replace(/convInfo\.outChannels/gi, ''+convInfo.outChannels)
        .replace(/padTop/gi, ''+padTop)
        .replace(/yRowStride/gi, ''+yRowStride)
        .replace(/filterHeight/gi, ''+filterHeight)
        .replace(/dilationHeight/gi, ''+dilationHeight)
        .replace(/filterStrides\[0\]/gi, ''+filterStrides[0])
        .replace(/xRowStride/gi, ''+xRowStride)
        .replace(/yColStride/gi, ''+yColStride)
        .replace(/padLeft/gi, ''+padLeft)
        .replace(/filterWidth/gi, ''+filterWidth)
        .replace(/dilationWidth/gi, ''+dilationWidth)
        .replace(/filterStrides\[1\]/gi, ''+filterStrides[1])
        .replace(/xColStride/gi, ''+xColStride)
        .replace(/xChannelStride/gi, ''+xChannelStride)
        .replace(/yChannelStride/gi, ''+yChannelStride)
        //.replace(/out/gi, 'this.out')
        //.replace(/weights/gi, 'this.weights')
        // .replace(/yRowStride/gi, ''+yRowStride)
        apply=new Function('data','out','weights', funcStr        )
        return {
            args:[out, weights],
            apply,
            shape,
            out
        }
    }
    Conv2d(weight: tf.Tensor<tf.Rank>[], params: any, predLayer: any, current: any) {
        

        let weights = weight[0].dataSync();
        let shape: number[] = predLayer.shape.slice(0);
        shape[shape.length - 1] = weight[0].shape[weight[0].shape.length - 1]

        let outLength = 1;
        shape.forEach(v => outLength *= v);


        const inpChannels = weight[0].shape[weight[0].shape.length - 2];
        const outChannels = shape[2]
        let out = new Float32Array(outLength)
        const kernelSize = params.kernelSize;

        let nWeights = new Float32Array(weights.length)
        let ci = 0;
        for (let c = 0; c < outChannels; c++) {
            for (let dy = 0; dy < kernelSize[0]; dy++) {
                for (let dx = 0; dx < kernelSize[1]; dx++) {
                    for (let z = 0; z < inpChannels; z++) {

                        nWeights[ci++] = weights[(dy * kernelSize[0] * inpChannels + dx * inpChannels + z) * outChannels + c];
                    }
                }
            }
        }
        let getLine = () => {
            let lines = [];
            for (let z = 0; z < inpChannels; z++) {
                lines.push(`val+= data[((y+dy)*${shape[1] + 2} + (x+dx))*${inpChannels}+${z}] * nWeights[ci++ ];`)
            }
            return lines.join('\n')
        }
        let getKernelX = () => {
            let lines = [];
            for (let dx = 0; dx < kernelSize[1]; dx++) {
                lines.push(getLine().replace(/dx/gi, '' + dx))
            }
            return lines.join('\n');
        }
        let getKernelY = () => {
            let lines = [];
            for (let dy = 0; dy < kernelSize[1]; dy++) {
                lines.push(getKernelX().replace(/dy/gi, '' + dy))
            }
            return lines.join('\n');
        }
        let levels = [
            new Function('data','out', 'nWeights', `
                if(out.length<${outLength}){
                    return ;
                }
                if(data.length<${inpChannels*shape[1]*shape[0]}){
                    return ;
                }
                for(let y =0;y<${shape[0]};y++){
                    for(let x =0;x<${shape[1]};x++){
                        let ci = 0;
                        for(let c =0;c<${outChannels};c++){
                            let val = 0;
                            ${getKernelY()}
                            out[(y*${shape[1]} + x)*${outChannels}+c] = val;
                        }
                    }
                }
                return out;
            `),
            new Function('data','out', 'nWeights', `
            if(out.length<${outLength}){
                return ;
            }
            if(data.length<${inpChannels*shape[1]*shape[0]}){
                return ;
            }
                for(let y =0;y<${shape[0]};y++){
                    for(let x =0;x<${shape[1]};x++){
                        let ci = 0;
                        for(let c =0;c<${outChannels};c++){
                            let val = 0;
                            for(let dy =0;dy<${kernelSize[1]};dy++){
                                ${getKernelX()}
                            }
                            out[(y*${shape[1]} + x)*${outChannels}+c] = val;
                        }
                    }
                }
                return out;
            `),
            new Function('data','out', 'nWeights', `
            if(out.length<${outLength}){
                return ;
            }
            if(data.length<${inpChannels*shape[1]*shape[0]}){
                return ;
            }
                for(let y =0;y<${shape[0]};++y){
                    for(let x =0;x<${shape[1]};x++){
                        let ci = 0;
                        for(let c =0;c<${outChannels};c++){
                            let val = 0;
                            for(let dy =0;dy<${kernelSize[1]};dy++){
                                for(let dx =0;dx<${kernelSize[1]};dx++){
                                    ${getLine()}
                                }
                            }
                            out[(y*${shape[1]} + x)*${outChannels}+c] = val;
                        }
                    }
                }
                return out;
            `),
        ]
        let levelsCount = [
            inpChannels * kernelSize[1] * kernelSize[0],
            inpChannels * kernelSize[1],
            //inpChannels,
            0,
        ]
        let index = levelsCount.findIndex(v => v < 50)
        return {
            args:[out, nWeights],
            apply: levels[index],
            // new Function('data', `
            // let out = this.out;
            // let nWeights = this.nWeights;
            //     for(let y =0;y<${shape[0]};y++){
            //         for(let x =0;x<${shape[1]};x++){
            //             let ci = 0;
            //             for(let c =0;c<${outChannels};c++){
            //                 let val = 0;
            //                 ${getKernelY()}
            //                 out[(y*${shape[1]} + x)*${outChannels}+c] = val;
            //             }
            //         }
            //     }
            //     return out;
            // `).bind({out,nWeights}),
            // apply:(data:any)=>{
            //     for(let y =0;y<shape[0];y++){
            //         for(let x =0;x<shape[1];x++){
            //                 let ci = 0;
            //             for(let c =0;c<outChannels;c++){
            //                 let val = 0;
            //                 for(let dy =0;dy<kernelSize[0];dy++){
            //                     for(let dx =0;dx<kernelSize[1];dx++){
            //                         for(let z =0;z<inpChannels;z++){
            //                             val+= data[((y+dy)*(shape[1]+2) + (x+dx))*inpChannels+z] * nWeights[ci++ ];
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
    BatchNormalization(weight: tf.Tensor<tf.Rank>[], params: any, predLayer?: any) {

        let weights = weight.map(v => v.dataSync());
        let shape: number[] = predLayer.shape.slice(0);
        shape[shape.length - 1] = weight[0].shape[weight[0].shape.length - 1]

        let outLength = 1;
        shape.forEach(v => outLength *= v);

        let out = new Float32Array(outLength)

        let sVals = weights[0];
        let offVals = weights[1];
        let mVals = weights[2];
        let varVals = weights[3];
        let varianceEpsilon = params.epsilon;
        const offValsLength = offVals.length;
        const sValsLength = sVals.length;

        //pre computed tensors
        varVals = varVals.map((v: number) => Math.sqrt(v + varianceEpsilon))

        sVals = sVals.map((v: number, i: number) => v / varVals[i])
        mVals = mVals.map((v: number, i: number) => v * sVals[i])
        offVals = offVals.map((v: number, i: number) => v - mVals[i])

        let getIfLine = (name: string, length: number) => {
            if (Math.log2(length) % 1 === 0) {
                let bits = length - 1;
                return `${name}=${name}&${bits};`
            } else {
                return `if (${name} >= ${length}) {
                    ${name} = 0;
                    }`
            }
        }

        return {
            args:[],
            apply: new Function('data', `
                let offi = 0;
                let si = 0;
                for (let i = 0; i < ${out.length}; i++) {

                    this.out[i] = this.offVals[offi++] + data[i] * this.sVals[si++];

                    ${getIfLine('offi', offValsLength)}

                    ${getIfLine('si', sValsLength)}
                }
                return this.out;
            `).bind({ sVals, offVals, out }),
            /*
            apply:(data:any)=>{
                let offi = 0;
                let mi = 0;
                let si = 0;
                let vi = 0;
                for (let i = 0; i < data.length; ++i) {
                    out[i] = offVals[offi++] +
                        (data[i] - mVals[mi++]) * sVals[si++] /
                            Math.sqrt(varVals[vi++] + varianceEpsilon);
                    if (offi >= offValsLength) {
                      offi = 0;
                    }
                    if (mi >= mValsLength) {
                      mi = 0;
                    }
                    if (si >= sValsLength) {
                      si = 0;
                    }
                    if (vi >= varValsLength) {
                      vi = 0;
                    }
                  }

                return out;
            },*/
            shape,
            out
        }
    }
}

export default Backend;