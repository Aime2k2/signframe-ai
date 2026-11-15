/** Small CPU implementation of the original SignLanguageCNN layers.
 * Weights are produced from the author's PyTorch checkpoint at build time.
 */
export class SignFrameCNN {
  constructor(manifest, binaryBuffer) {
    if (manifest.format !== 'signframe-cnn-f32-v1') throw Error('Unexpected CNN model format');
    if (binaryBuffer.byteLength % 4 !== 0) throw Error('Corrupt CNN weights');
    const all = new Float32Array(binaryBuffer);
    const layer = name => {
      const entry = manifest.layers[name];
      if (!entry || entry.offset + entry.count > all.length) throw Error('Bad tensor ' + name);
      return all.subarray(entry.offset, entry.offset + entry.count);
    };
    this.k1 = layer('features.0.weight'); this.b1 = layer('features.0.bias');
    this.k2 = layer('features.3.weight'); this.b2 = layer('features.3.bias');
    this.k3 = layer('classifier.1.weight'); this.b3 = layer('classifier.1.bias');
    this.k4 = layer('classifier.4.weight'); this.b4 = layer('classifier.4.bias');
    this.labels = manifest.labels;
  }
  static async load(base = './model/') {
    const [metadata, weights] = await Promise.all([
      fetch(base + 'weights.json'), fetch(base + 'weights.bin')
    ]);
    if (!metadata.ok || !weights.ok) throw Error('CNN files not available; check the GitHub Pages build');
    return new SignFrameCNN(await metadata.json(), await weights.arrayBuffer());
  }
  convRelu(input, inC, spatial, outC, filters, bias) {
    const out = new Float32Array(outC * spatial * spatial);
    const plane = spatial * spatial;
    for (let oc = 0; oc < outC; oc++) {
      const dest = oc * plane, filterBase = oc * inC * 9;
      for (let y = 0; y < spatial; y++) for (let x = 0; x < spatial; x++) {
        let sum = bias[oc];
        for (let ic = 0; ic < inC; ic++) {
          const base = ic * plane, fbase = filterBase + ic * 9;
          for (let ky = 0; ky < 3; ky++) {
            const iy = y + ky - 1;
            if (iy < 0 || iy >= spatial) continue;
            for (let kx = 0; kx < 3; kx++) {
              const ix = x + kx - 1;
              if (ix < 0 || ix >= spatial) continue;
              sum += input[base + iy * spatial + ix] * filters[fbase + ky * 3 + kx];
            }
          }
        }
        out[dest + y * spatial + x] = Math.max(0, sum);
      }
    }
    return out;
  }
  maxPool2(input, channels, spatial) {
    const next = spatial / 2, out = new Float32Array(channels * next * next);
    for (let c = 0; c < channels; c++) {
      const p = c * spatial * spatial, q = c * next * next;
      for (let y = 0; y < next; y++) for (let x = 0; x < next; x++) {
        const i = p + 2*y*spatial + 2*x;
        out[q + y*next + x] = Math.max(input[i],input[i+1],input[i+spatial],input[i+spatial+1]);
      }
    }
    return out;
  }
  dense(input, outC, weights, biases, relu = false) {
    const out = new Float32Array(outC);
    for (let j = 0; j < outC; j++) {
      let v = biases[j], offset = j*input.length;
      for (let i = 0; i < input.length; i++) v += input[i]*weights[offset+i];
      out[j] = relu ? Math.max(0,v) : v;
    }
    return out;
  }
  predict(gray28) {
    if (!gray28 || gray28.length !== 784) throw Error('Expected a 28x28 grayscale image');
    const p1 = this.maxPool2(this.convRelu(gray28,1,28,32,this.k1,this.b1),32,28);
    const p2 = this.maxPool2(this.convRelu(p1,32,14,64,this.k2,this.b2),64,14);
    const h = this.dense(p2,128,this.k3,this.b3,true);
    const logits = this.dense(h,24,this.k4,this.b4,false);
    const mx = Math.max(...logits), exps = logits.map(v=>Math.exp(v-mx));
    const total = exps.reduce((s,v)=>s+v,0);
    let iBest = 0;
    for (let i=1;i<logits.length;i++) if(logits[i]>logits[iBest]) iBest=i;
    return {letter:this.labels[iBest],confidence:exps[iBest]/total};
  }
}
