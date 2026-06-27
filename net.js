/**
 * NeuralNetwork - 手写数字识别神经网络
 * 架构: 784 → 128 (ReLU) → 64 (ReLU) → 10 (Softmax)
 *
 * 权重存储: W_shape = [fan_out, fan_in]
 *   每行对应一个输出神经元的权重
 *   前向: output = matmul(W, input) + b
 *
 * 优化: SGD with Momentum (0.9) + L2 Weight Decay (0.0001)
 */

class NeuralNetwork {
  constructor(lr = 0.1) {
    this.lr = lr;
    this.momentum = 0.0;  // 多分类场景 momentum 弊大于利，暂不使用
    this.trainingCount = 0;
    this.initParams();
  }

  /** Xavier/Glorot 初始化 */
  initParams() {
    this.W1 = randn([128, 784], 128, 784);
    this.b1 = zeros(128);
    this.W2 = randn([64, 128], 64, 128);
    this.b2 = zeros(64);
    this.W3 = randn([10, 64], 10, 64);
    this.b3 = zeros(10);
    this.resetMomentum();
  }

  resetMomentum() {
    this.vW1 = zerosMat(128, 784);
    this.vb1 = zeros(128);
    this.vW2 = zerosMat(64, 128);
    this.vb2 = zeros(64);
    this.vW3 = zerosMat(10, 64);
    this.vb3 = zeros(10);
  }

  /** 随机初始化所有权重（重新开始训练时调用） */
  randomize() {
    this.initParams();
    this.trainingCount = 0;
  }

  // ── 前向传播 ────────────────────────────────────────

  forward(input) {
    if (input.length !== 784) throw new Error(`期望输入长度 784，收到 ${input.length}`);

    // 隐藏层 1: 784 → 128, ReLU
    const z1 = addVec(matmul(this.W1, input), this.b1);
    const a1 = relu(z1);

    // 隐藏层 2: 128 → 64, ReLU
    const z2 = addVec(matmul(this.W2, a1), this.b2);
    const a2 = relu(z2);

    // 输出层: 64 → 10, Softmax
    const z3 = addVec(matmul(this.W3, a2), this.b3);
    const output = softmax(z3);

    return { output, cache: { z1, a1, z2, a2, z3, input } };
  }

  /** 预测: 返回概率和预测类别 */
  predict(input) {
    const { output } = this.forward(input);
    let predictedClass = 0;
    let maxProb = output[0];
    for (let i = 1; i < output.length; i++) {
      if (output[i] > maxProb) {
        maxProb = output[i];
        predictedClass = i;
      }
    }
    return { output, predictedClass };
  }

  // ── 训练 ────────────────────────────────────────────

  train(input, label) {
    // 前向传播 (用当前权重)
    const { output, cache } = this.forward(input);

    // one-hot 编码
    const target = zeros(10);
    target[label] = 1;

    // 损失 (交叉熵)
    const loss = crossEntropy(output, target);

    // === 反向传播 ===
    // 输出层梯度: softmax + cross-entropy 合并公式
    const dz3 = subVec(output, target);           // [10]
    const dW3 = outer(dz3, cache.a2);             // [10, 64]
    const db3 = dz3.slice();                      // [10]

    // 隐藏层 2
    const da2 = matmulT(this.W3, dz3);            // W3^T * dz3 → [64]
    const dz2 = mulElem(da2, reluDeriv(cache.z2)); // [64]
    const dW2 = outer(dz2, cache.a1);             // [64, 128]
    const db2 = dz2.slice();                      // [64]

    // 隐藏层 1
    const da1 = matmulT(this.W2, dz2);            // W2^T * dz2 → [128]
    const dz1 = mulElem(da1, reluDeriv(cache.z1)); // [128]
    const dW1 = outer(dz1, cache.input);          // [128, 784]
    const db1 = dz1.slice();                      // [128]

    // === SGD with Momentum + L2 ===
    const mu = this.momentum;
    const wd = 0.0001;

    // 输出层: v = mu*v + lr*dW + lr*wd*W, W = W - v
    this.vW3 = addMat(scaleMat(this.vW3, mu), scaleMat(dW3, -this.lr));
    addMatInPlace(this.vW3, scaleMat(this.W3, -this.lr * wd));
    this.W3 = addMat(this.W3, this.vW3);
    this.b3 = addArr(this.b3, scaleArr(db3, -this.lr));

    // 隐藏层 2
    this.vW2 = addMat(scaleMat(this.vW2, mu), scaleMat(dW2, -this.lr));
    addMatInPlace(this.vW2, scaleMat(this.W2, -this.lr * wd));
    this.W2 = addMat(this.W2, this.vW2);
    this.b2 = addArr(this.b2, scaleArr(db2, -this.lr));

    // 隐藏层 1
    this.vW1 = addMat(scaleMat(this.vW1, mu), scaleMat(dW1, -this.lr));
    addMatInPlace(this.vW1, scaleMat(this.W1, -this.lr * wd));
    this.W1 = addMat(this.W1, this.vW1);
    this.b1 = addArr(this.b1, scaleArr(db1, -this.lr));

    this.trainingCount++;
    return { loss, output };
  }

  /**
   * Mini-batch 训练: 一批样本的梯度求和后取平均，一次性更新
   * @param {Array<{input: Float64Array, label: number}>} batch
   * @returns {{ loss: number, accuracy: number }}
   */
  trainBatch(batch) {
    const size = batch.length;
    if (size === 0) return { loss: 0, accuracy: 0 };

    // 初始化累计梯度
    let dW3_acc = zerosMat(10, 64);
    let db3_acc = zeros(10);
    let dW2_acc = zerosMat(64, 128);
    let db2_acc = zeros(64);
    let dW1_acc = zerosMat(128, 784);
    let db1_acc = zeros(128);

    let totalLoss = 0;
    let correct = 0;

    for (const { input, label } of batch) {
      const { output, cache } = this.forward(input);

      // 统计
      const pred = argmax(output);
      if (pred === label) correct++;
      const target = zeros(10);
      target[label] = 1;
      totalLoss += crossEntropy(output, target);

      // 输出层梯度
      const dz3 = subVec(output, target);
      addMatInPlace(dW3_acc, outer(dz3, cache.a2));
      addArrInPlace(db3_acc, dz3);

      // 隐藏层 2
      const da2 = matmulT(this.W3, dz3);
      const dz2 = mulElem(da2, reluDeriv(cache.z2));
      addMatInPlace(dW2_acc, outer(dz2, cache.a1));
      addArrInPlace(db2_acc, dz2);

      // 隐藏层 1
      const da1 = matmulT(this.W2, dz2);
      const dz1 = mulElem(da1, reluDeriv(cache.z1));
      addMatInPlace(dW1_acc, outer(dz1, cache.input));
      addArrInPlace(db1_acc, dz1);
    }

    // 取平均梯度
    const invSize = 1 / size;
    dW3_acc = scaleMat(dW3_acc, invSize);
    db3_acc = scaleArr(db3_acc, invSize);
    dW2_acc = scaleMat(dW2_acc, invSize);
    db2_acc = scaleArr(db2_acc, invSize);
    dW1_acc = scaleMat(dW1_acc, invSize);
    db1_acc = scaleArr(db1_acc, invSize);

    // 一次性更新权重 (SGD + L2)
    const wd = 0.0001;
    this.W3 = subMat(this.W3, scaleMat(dW3_acc, this.lr));
    addMatInPlace(this.W3, scaleMat(this.W3, -this.lr * wd));
    this.b3 = subArr(this.b3, scaleArr(db3_acc, this.lr));

    this.W2 = subMat(this.W2, scaleMat(dW2_acc, this.lr));
    addMatInPlace(this.W2, scaleMat(this.W2, -this.lr * wd));
    this.b2 = subArr(this.b2, scaleArr(db2_acc, this.lr));

    this.W1 = subMat(this.W1, scaleMat(dW1_acc, this.lr));
    addMatInPlace(this.W1, scaleMat(this.W1, -this.lr * wd));
    this.b1 = subArr(this.b1, scaleArr(db1_acc, this.lr));

    this.trainingCount += size;
    return { loss: totalLoss / size, accuracy: correct / size };
  }

  // ── 序列化 ──────────────────────────────────────────

  toJSON() {
    return {
      version: 1,
      architecture: [784, 128, 64, 10],
      activations: ['linear', 'relu', 'relu', 'softmax'],
      weights: {
        W1: flatten2D(this.W1),
        b1: Array.from(this.b1),
        W2: flatten2D(this.W2),
        b2: Array.from(this.b2),
        W3: flatten2D(this.W3),
        b3: Array.from(this.b3),
      },
      shapes: {
        W1: [128, 784], b1: [128],
        W2: [64, 128],  b2: [64],
        W3: [10, 64],   b3: [10],
      },
      metadata: {
        description: '手写数字识别神经网络',
        date: new Date().toISOString(),
        trainingSamples: this.trainingCount,
        learningRate: this.lr,
      },
    };
  }

  static fromJSON(data) {
    if (!data || !data.weights || !data.shapes) {
      throw new Error('无效的模型文件');
    }
    const nn = new NeuralNetwork(data.metadata?.learningRate || 0.05);
    nn.W1 = reshape2D(data.weights.W1, data.shapes.W1[0], data.shapes.W1[1]);
    nn.b1 = new Float64Array(data.weights.b1);
    nn.W2 = reshape2D(data.weights.W2, data.shapes.W2[0], data.shapes.W2[1]);
    nn.b2 = new Float64Array(data.weights.b2);
    nn.W3 = reshape2D(data.weights.W3, data.shapes.W3[0], data.shapes.W3[1]);
    nn.b3 = new Float64Array(data.weights.b3);
    nn.resetMomentum();
    nn.trainingCount = data.metadata?.trainingSamples || 0;
    nn.lr = data.metadata?.learningRate || 0.05;
    return nn;
  }
}


// ═══════════════════════════════════════════════════════
//  矩阵 / 向量 工具函数
// ═══════════════════════════════════════════════════════

/** 创建 [rows x cols] 的随机矩阵 (Xavier初始化) */
function randn([rows, cols], fanIn, fanOut) {
  const scale = Math.sqrt(2 / (fanIn + fanOut));
  const m = new Array(rows);
  for (let i = 0; i < rows; i++) {
    const row = new Float64Array(cols);
    for (let j = 0; j < cols; j++) {
      row[j] = gaussRandom() * scale;
    }
    m[i] = row;
  }
  return m;
}

/** 创建长度为 n 的零向量 */
function zeros(n) {
  return new Float64Array(n);
}

/** 创建零矩阵 */
function zerosMat(rows, cols) {
  const result = new Array(rows);
  for (let i = 0; i < rows; i++) {
    result[i] = new Float64Array(cols);
  }
  return result;
}

/** 高斯随机数 (Box-Muller) */
function gaussRandom() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** 矩阵乘法: mat * vec (矩阵 × 列向量) */
function matmul(mat, vec) {
  const rows = mat.length;
  const cols = mat[0].length;
  const result = new Float64Array(rows);
  for (let i = 0; i < rows; i++) {
    let sum = 0;
    const row = mat[i];
    for (let j = 0; j < cols; j++) {
      sum += row[j] * vec[j];
    }
    result[i] = sum;
  }
  return result;
}

/** 矩阵转置乘法: M^T * vec */
function matmulT(mat, vec) {
  const rows = mat.length;
  const cols = mat[0].length;
  const result = new Float64Array(cols);
  for (let j = 0; j < cols; j++) {
    let sum = 0;
    for (let i = 0; i < rows; i++) {
      sum += mat[i][j] * vec[i];
    }
    result[j] = sum;
  }
  return result;
}

/** 向量加法: 对矩阵的一行加偏置 */
function addVec(matResult, bias) {
  const result = new Float64Array(matResult.length);
  for (let i = 0; i < matResult.length; i++) {
    result[i] = matResult[i] + bias[i];
  }
  return result;
}

/** 向量减法: a - b */
function subVec(a, b) {
  const result = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) {
    result[i] = a[i] - b[i];
  }
  return result;
}

/** 元素乘: a * b (逐元素) */
function mulElem(a, b) {
  const result = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) {
    result[i] = a[i] * b[i];
  }
  return result;
}

/** 外积: u ⊗ v → [u.length x v.length] */
function outer(u, v) {
  const rows = u.length;
  const cols = v.length;
  const result = new Array(rows);
  for (let i = 0; i < rows; i++) {
    const row = new Float64Array(cols);
    const ui = u[i];
    for (let j = 0; j < cols; j++) {
      row[j] = ui * v[j];
    }
    result[i] = row;
  }
  return result;
}

/** 缩放向量: scalar * vec */
function scaleArr(arr, scalar) {
  const result = new Float64Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    result[i] = arr[i] * scalar;
  }
  return result;
}

/** 缩放矩阵: scalar * mat */
function scaleMat(mat, scalar) {
  const rows = mat.length;
  const cols = mat[0].length;
  const result = new Array(rows);
  for (let i = 0; i < rows; i++) {
    const row = new Float64Array(cols);
    const srcRow = mat[i];
    for (let j = 0; j < cols; j++) {
      row[j] = srcRow[j] * scalar;
    }
    result[i] = row;
  }
  return result;
}

/** 矩阵加法: a + b (返回新矩阵) */
function addMat(a, b) {
  const rows = a.length;
  const cols = a[0].length;
  const result = new Array(rows);
  for (let i = 0; i < rows; i++) {
    const row = new Float64Array(cols);
    const aRow = a[i];
    const bRow = b[i];
    for (let j = 0; j < cols; j++) {
      row[j] = aRow[j] + bRow[j];
    }
    result[i] = row;
  }
  return result;
}

/** 矩阵加法 in-place: a += b (直接修改 a) */
function addMatInPlace(a, b) {
  const rows = a.length;
  const cols = a[0].length;
  for (let i = 0; i < rows; i++) {
    const aRow = a[i];
    const bRow = b[i];
    for (let j = 0; j < cols; j++) {
      aRow[j] += bRow[j];
    }
  }
}

/** 向量加法 in-place: a += b */
function addArrInPlace(a, b) {
  for (let i = 0; i < a.length; i++) {
    a[i] += b[i];
  }
}

/** 矩阵减法: a - b (逐元素) */
function subMat(a, b) {
  const rows = a.length;
  const cols = a[0].length;
  const result = new Array(rows);
  for (let i = 0; i < rows; i++) {
    const row = new Float64Array(cols);
    const aRow = a[i];
    const bRow = b[i];
    for (let j = 0; j < cols; j++) {
      row[j] = aRow[j] - bRow[j];
    }
    result[i] = row;
  }
  return result;
}

/** 向量减法: a - b (数组版本) */
function subArr(a, b) {
  const result = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) {
    result[i] = a[i] - b[i];
  }
  return result;
}

/** 向量加法: a + b (数组版本) */
function addArr(a, b) {
  const result = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) {
    result[i] = a[i] + b[i];
  }
  return result;
}

// ── 激活函数 ────────────────────────────────────────

function relu(x) {
  const result = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) {
    result[i] = Math.max(0, x[i]);
  }
  return result;
}

function reluDeriv(x) {
  const result = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) {
    result[i] = x[i] > 0 ? 1 : 0;
  }
  return result;
}

function softmax(z) {
  let maxZ = z[0];
  for (let i = 1; i < z.length; i++) {
    if (z[i] > maxZ) maxZ = z[i];
  }
  const exps = new Float64Array(z.length);
  let sumExp = 0;
  for (let i = 0; i < z.length; i++) {
    exps[i] = Math.exp(z[i] - maxZ);
    sumExp += exps[i];
  }
  const result = new Float64Array(z.length);
  for (let i = 0; i < z.length; i++) {
    result[i] = exps[i] / sumExp;
  }
  return result;
}

// ── 损失函数 ─────────────────────────────────────────

function crossEntropy(output, target) {
  const eps = 1e-10;
  let loss = 0;
  for (let i = 0; i < output.length; i++) {
    loss -= target[i] * Math.log(Math.max(output[i], eps));
  }
  return loss;
}

// ── 序列化辅助 ──────────────────────────────────────

function flatten2D(mat) {
  const result = [];
  for (let i = 0; i < mat.length; i++) {
    for (let j = 0; j < mat[i].length; j++) {
      result.push(mat[i][j]);
    }
  }
  return result;
}

function reshape2D(arr, rows, cols) {
  const result = new Array(rows);
  for (let i = 0; i < rows; i++) {
    const row = new Float64Array(cols);
    for (let j = 0; j < cols; j++) {
      row[j] = arr[i * cols + j];
    }
    result[i] = row;
  }
  return result;
}