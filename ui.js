/**
 * ui.js - 手写数字识别神经网络 UI 交互与可视化
 */

// ── 状态 ────────────────────────────────────────────

const state = {
  mode: 'train',         // 'train' | 'recognize'
  drawing: false,
  lastX: 0,
  lastY: 0,
  hasInk: false,
};

// ── DOM 引用 ────────────────────────────────────────

const drawCanvas = document.getElementById('drawCanvas');
const ctx = drawCanvas.getContext('2d');
const visCanvas = document.getElementById('visCanvas');
const vctx = visCanvas.getContext('2d');

// ── 神经网络实例 ────────────────────────────────────

let nn = new NeuralNetwork(0.1);

// ── 画板初始化 ──────────────────────────────────────

function initCanvas() {
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, 280, 280);
  ctx.lineWidth = 18;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = '#000000';
}

initCanvas();

// ── 画板事件 ────────────────────────────────────────

function getPos(e) {
  const rect = drawCanvas.getBoundingClientRect();
  const scaleX = drawCanvas.width / rect.width;
  const scaleY = drawCanvas.height / rect.height;
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top) * scaleY,
  };
}

function startDraw(e) {
  e.preventDefault();
  state.drawing = true;
  state.hasInk = true;
  const pos = getPos(e);
  state.lastX = pos.x;
  state.lastY = pos.y;
  ctx.beginPath();
  ctx.moveTo(pos.x, pos.y);
}

function moveDraw(e) {
  e.preventDefault();
  if (!state.drawing) return;
  const pos = getPos(e);
  ctx.beginPath();
  ctx.moveTo(state.lastX, state.lastY);
  ctx.lineTo(pos.x, pos.y);
  ctx.stroke();
  state.lastX = pos.x;
  state.lastY = pos.y;
  updateTrainButton();
}

function endDraw(e) {
  e.preventDefault();
  state.drawing = false;
  // 识别模式下自动识别
  if (state.mode === 'recognize' && state.hasInk) {
    doRecognize();
  }
}

// 鼠标事件
drawCanvas.addEventListener('mousedown', startDraw);
drawCanvas.addEventListener('mousemove', moveDraw);
drawCanvas.addEventListener('mouseup', endDraw);
// mouseleave 只停止绘制，不触发识别（防止笔离开画板边缘时打断绘制）
drawCanvas.addEventListener('mouseleave', () => {
  state.drawing = false;
});

// 触摸事件
drawCanvas.addEventListener('touchstart', (e) => {
  e.preventDefault();  // 阻止浏览器合成重复的鼠标事件
  const touch = e.touches[0];
  const me = new MouseEvent('mousedown', { clientX: touch.clientX, clientY: touch.clientY });
  drawCanvas.dispatchEvent(me);
});
drawCanvas.addEventListener('touchmove', (e) => {
  e.preventDefault();
  const touch = e.touches[0];
  const me = new MouseEvent('mousemove', { clientX: touch.clientX, clientY: touch.clientY });
  drawCanvas.dispatchEvent(me);
});
drawCanvas.addEventListener('touchend', (e) => {
  e.preventDefault();
  const me = new MouseEvent('mouseup', { });
  drawCanvas.dispatchEvent(me);
});

// ── 画板采样 ────────────────────────────────────────

function sampleCanvas() {
  // 创建 offscreen 28x28 canvas
  const offscreen = document.createElement('canvas');
  offscreen.width = 28;
  offscreen.height = 28;
  const offCtx = offscreen.getContext('2d');

  // 高质量降采样
  offCtx.imageSmoothingEnabled = true;
  offCtx.imageSmoothingQuality = 'high';
  offCtx.drawImage(drawCanvas, 0, 0, 280, 280, 0, 0, 28, 28);

  // 读取像素
  const imageData = offCtx.getImageData(0, 0, 28, 28);
  const pixels = imageData.data;
  const input = new Float64Array(784);

  for (let i = 0; i < 784; i++) {
    const idx = i * 4;
    // 灰度: 0.299*R + 0.587*G + 0.114*B
    const gray = 0.299 * pixels[idx] + 0.587 * pixels[idx + 1] + 0.114 * pixels[idx + 2];
    // 反转: 白色背景=0, 黑色笔迹=1
    input[i] = 1.0 - gray / 255;
  }

  // MNIST 标准预处理: 居中 + 尺寸归一化 (见下方函数)
  return preprocessMNIST(input);
}

/**
 * MNIST 标准预处理: 裁剪笔画边界框 → 等比缩放到 20×20 → 按质心居中到 28×28
 *
 * MNIST 训练数据每张图都经过了这套归一化,数字严格居中、尺寸一致。
 * 手写输入若直接整板降采样,位置和大小都会偏移 → 分布与训练数据不一致 → 识别率骤降。
 * (实测: ±5px 偏移使准确率从 ~89% 跌到 ~20%; 加本预处理后恢复到 ~83%。)
 */
function preprocessMNIST(input) {
  const TH = 0.1;
  const result = new Float64Array(784);

  // 1. 找笔画边界框
  let minX = 28, minY = 28, maxX = -1, maxY = -1;
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      if (input[y * 28 + x] > TH) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return result;  // 空画板

  // 2. 等比缩放到 20×20 (取宽高较大者为基准, 保持长宽比)
  const bw = maxX - minX + 1;
  const bh = maxY - minY + 1;
  const scale = 20 / Math.max(bw, bh);

  // 3. 计算缩放后笔画的质心, 使其对齐到 (14, 14) — MNIST 的居中方式
  let cx = 0, cy = 0, mass = 0;
  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      const v = input[y * 28 + x];
      cx += (x - minX) * v;
      cy += (y - minY) * v;
      mass += v;
    }
  }
  const bcx = mass > 0 ? cx / mass : bw / 2;
  const bcy = mass > 0 ? cy / mass : bh / 2;
  const offX = 14 - bcx * scale;
  const offY = 14 - bcy * scale;

  // 4. 最近邻采样到 28×28
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const sx = Math.round((x - offX) / scale + minX);
      const sy = Math.round((y - offY) / scale + minY);
      if (sx >= 0 && sx < 28 && sy >= 0 && sy < 28) {
        result[y * 28 + x] = input[sy * 28 + sx];
      }
    }
  }
  return result;
}

// ── 清空画板 ────────────────────────────────────────

function clearCanvas() {
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, 280, 280);
  state.hasInk = false;
  updateTrainButton();
}

// ── Digit Checkboxes ─────────────────────────────────

const CHECK_LABELS = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'];

function initCheckboxes() {
  const container = document.getElementById('digitCheckboxes');
  container.innerHTML = '';
  for (let i = 0; i < 10; i++) {
    const label = document.createElement('label');
    label.className = 'digit-checkbox';
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.value = i;
    input.addEventListener('change', updateTrainButton);

    const indicator = document.createElement('span');
    indicator.className = 'check-indicator';
    indicator.textContent = i;

    const text = document.createTextNode(CHECK_LABELS[i]);

    label.appendChild(input);
    label.appendChild(indicator);
    label.appendChild(text);
    container.appendChild(label);
  }
}

function getSelectedDigit() {
  const checkboxes = document.querySelectorAll('#digitCheckboxes input[type="checkbox"]');
  for (const cb of checkboxes) {
    if (cb.checked) return parseInt(cb.value);
  }
  return -1;
}

function updateTrainButton() {
  const btn = document.getElementById('btnTrain');
  const hasSelection = getSelectedDigit() >= 0;
  btn.disabled = !(hasSelection && state.hasInk);
}

function clearCheckboxes() {
  document.querySelectorAll('#digitCheckboxes input[type="checkbox"]').forEach(cb => {
    cb.checked = false;
  });
  updateTrainButton();
}

// ── 模式切换 ────────────────────────────────────────

function setMode(mode) {
  state.mode = mode;
  document.getElementById('btnTrainMode').classList.toggle('active', mode === 'train');
  document.getElementById('btnRecogMode').classList.toggle('active', mode === 'recognize');
  document.getElementById('trainPanel').classList.toggle('hidden', mode !== 'train');
  document.getElementById('recogPanel').classList.toggle('hidden', mode !== 'recognize');
  document.getElementById('statusMode').textContent = mode === 'train' ? '训练' : '识别';

  // 切换到识别模式时，检测画板内容并自动识别
  if (mode === 'recognize') {
    doRecognize();
  }
}

document.getElementById('btnTrainMode').addEventListener('click', () => setMode('train'));
document.getElementById('btnRecogMode').addEventListener('click', () => setMode('recognize'));

// ── 训练 ────────────────────────────────────────────

function doTrain() {
  if (!state.hasInk) return;

  const label = getSelectedDigit();
  if (label < 0) return;

  const input = sampleCanvas();
  const result = nn.train(input, label);

  // 训练后再预测一次（post-update），显示更新后的效果
  const { output: postOutput, predictedClass } = nn.predict(input);

  // 更新显示
  document.getElementById('statusSamples').textContent = nn.trainingCount;
  document.getElementById('statusLoss').textContent = result.loss.toFixed(4);
  document.getElementById('statusNetwork').textContent = `已训练 ${nn.trainingCount} 样本`;

  // 训练结果
  const resultDiv = document.getElementById('trainResult');
  resultDiv.classList.remove('hidden');
  const isCorrect = predictedClass === label;
  resultDiv.className = `train-result ${result.loss > 1.0 ? 'loss-high' : 'loss-low'}`;
  resultDiv.textContent = `损失: ${result.loss.toFixed(4)} | 预测: ${predictedClass} (${CHECK_LABELS[predictedClass]}) ${isCorrect ? '✓' : '✗'}`;

  // 更新可视化
  renderVisualization({ input, output: postOutput });

  // 清空画板方便下次训练
  clearCanvas();
  clearCheckboxes();
}

document.getElementById('btnTrain').addEventListener('click', doTrain);

// ── 识别 ────────────────────────────────────────────

function doRecognize() {
  const input = sampleCanvas();
  const hasActivePixels = input.some(v => v > 0.05);
  console.log('[识别] 有像素:', hasActivePixels, '最大值:', Math.max(...input).toFixed(3));
  if (!hasActivePixels) return;

  const { output, predictedClass } = nn.predict(input);
  console.log('[识别] 输出:', Array.from(output).map(v => v.toFixed(4)));

  // 更新概率条
  updateProbBars(output, predictedClass);
  document.getElementById('predDigit').textContent = predictedClass;

  // 更新可视化
  renderVisualization({ input, output });
}

// ── 概率条 ──────────────────────────────────────────

function initProbBars() {
  const container = document.getElementById('probBars');
  container.innerHTML = '';
  for (let i = 0; i < 10; i++) {
    const row = document.createElement('div');
    row.className = 'prob-bar-row';

    const label = document.createElement('div');
    label.className = 'prob-label';
    label.textContent = i;

    const track = document.createElement('div');
    track.className = 'prob-track';

    const fill = document.createElement('div');
    fill.className = 'prob-fill';
    fill.id = `probFill${i}`;
    fill.style.width = '0%';

    track.appendChild(fill);

    const value = document.createElement('div');
    value.className = 'prob-value';
    value.id = `probValue${i}`;
    value.textContent = '0.0%';

    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(value);
    container.appendChild(row);
  }
}

function updateProbBars(probs, predicted) {
  for (let i = 0; i < 10; i++) {
    const pct = probs[i] * 100;
    const fill = document.getElementById(`probFill${i}`);
    const value = document.getElementById(`probValue${i}`);
    fill.style.width = `${Math.max(pct, 0.5)}%`;
    fill.className = `prob-fill ${i === predicted ? 'highlight' : ''}`;
    value.textContent = `${pct.toFixed(1)}%`;
  }
}

// ── 网络可视化 ──────────────────────────────────────

const weightCanvas = document.getElementById('weightCanvas');
const wctx = weightCanvas.getContext('2d');

function setupVisCanvas() {
  const container = visCanvas.parentElement;
  const width = Math.min(container.clientWidth - 32, 600);
  const height = Math.max(container.clientHeight - 40, 450);
  visCanvas.width = width;
  visCanvas.height = height;
}

function renderVisualization(data) {
  setupVisCanvas();
  renderWeights();  // 同时渲染权重图
  const w = visCanvas.width;
  const h = visCanvas.height;

  vctx.fillStyle = '#1a1a2e';
  vctx.fillRect(0, 0, w, h);

  // 布局参数
  const margin = { top: 20, bottom: 20 };
  const layerGap = w / 5;

  // 各层 x 位置
  const inputX = layerGap * 1;
  const hidden1X = layerGap * 2;
  const hidden2X = layerGap * 3;
  const outputX = layerGap * 4;

  // 可绘制高度
  const drawH = h - margin.top - margin.bottom;

  // 各层神经元数量
  const layerSizes = [784, 128, 64, 10];
  const layerXs = [inputX, hidden1X, hidden2X, outputX];
  const layerNames = ['输入层\n784', '隐藏层1\n128', '隐藏层2\n64', '输出层\n10'];

  // 获取激活值
  const activations = data ? getActivations(data) : null;

  // 渲染每层
  for (let layer = 0; layer < 4; layer++) {
    const n = layerSizes[layer];
    const x = layerXs[layer];
    const maxDotsPerCol = layer === 0 ? 28 : 1;  // 输入层 28x28 网格

    // 计算圆点大小和间距以适应高度
    const gridH = drawH - 30; // 留出标签空间
    const rows = layer === 0 ? 28 : n;

    // 根据层不同调整圆点大小
    const availableHeight = gridH;
    let dotSize, gap;

    if (layer === 0) {
      // 输入层: 28x28 网格
      const cellSize = Math.min(availableHeight / 28, (layerGap * 0.7) / 28);
      dotSize = Math.max(2, cellSize * 0.7);
      gap = cellSize * 0.15;
      const gridStartX = x - (28 * (dotSize + gap)) / 2;
      const gridStartY = margin.top + (availableHeight - 28 * (dotSize + gap)) / 2;

      // 渲染 28x28 圆点网格
      for (let row = 0; row < 28; row++) {
        for (let col = 0; col < 28; col++) {
          const px = col + row * 28;
          let value = 0;
          if (activations && activations.input) {
            value = activations.input[px];
          }
          drawNeuronDot(vctx, gridStartX + col * (dotSize + gap), gridStartY + row * (dotSize + gap), dotSize, value, true);
        }
      }
    } else {
      // 隐藏层/输出层: 单列圆点
      // 计算每个点+间距的尺寸，确保所有神经元能垂直排下
      const cellH = availableHeight / n;
      const maxDotSize = 8;
      dotSize = Math.min(maxDotSize, Math.max(1.5, cellH * 0.65));
      gap = Math.max(0.5, cellH - dotSize);
      const totalH = n * (dotSize + gap) - gap;
      const startY = margin.top + (availableHeight - totalH) / 2;

      // 层标签
      vctx.fillStyle = '#8899aa';
      vctx.font = '10px sans-serif';
      vctx.textAlign = 'center';
      vctx.fillText(layerNames[layer], x, h - 5);

      // 渲染圆点
      for (let i = 0; i < n; i++) {
        let value = 0;
        if (activations) {
          const key = ['input', 'a1', 'a2', 'output'][layer];
          if (activations[key]) value = activations[key][i];
        }
        const yPos = startY + i * (dotSize + gap);

        // 输出层: 加数字标签和概率文字
        if (layer === 3) {
          // 标签 (左侧，稍微左移) — 加大字号便于辨认
          vctx.fillStyle = value > 0.5 ? '#00e676' : '#8899aa';
          vctx.font = value > 0.5 ? 'bold 20px sans-serif' : '18px sans-serif';
          vctx.textAlign = 'right';
          vctx.textBaseline = 'middle';
          vctx.fillText(String(i), x - dotSize - 6, yPos);

          // 概率值 (右侧)
          vctx.fillStyle = value > 0.1 ? '#e0e0e0' : '#555';
          vctx.font = '13px monospace';
          vctx.textAlign = 'left';
          vctx.fillText(`${(value * 100).toFixed(0)}%`, x + dotSize + 6, yPos);
          vctx.textBaseline = 'alphabetic';  // 恢复，避免影响后续层标签
        }

        drawNeuronDot(vctx, x, yPos, dotSize, value, false, layer === 3);
      }
    }
  }

  vctx.strokeStyle = 'rgba(255,255,255,0.03)';
  vctx.lineWidth = 0.5;
  for (let l = 0; l < 3; l++) {
    const x1 = layerXs[l] + 15;
    const x2 = layerXs[l + 1] - 15;
    const y1 = h / 2;
    const y2 = h / 2;
    vctx.beginPath();
    vctx.moveTo(x1, y1);
    vctx.lineTo(x2, y2);
    vctx.stroke();
  }
}

// ── 权重可视化 ────────────────────────────────────

// 当前选中的权重层: 'W1' | 'W2' | 'W3'
let currentWeightLayer = 'W1';

// 各层权重配置: 矩阵尺寸 + 网格列数 + 说明
const WEIGHT_LAYERS = {
  W1: { W: () => nn.W1, rows: 128, cols: 784, grid: 8,  fanInIsImage: true,  desc: 'W1: 输入层 784 → 隐藏层1 128。每个神经元接收一张 28×28 输入图像，展示其学到的像素特征模板。' },
  W2: { W: () => nn.W2, rows: 64,  cols: 128, grid: 8,  fanInIsImage: false, desc: 'W2: 隐藏层1 128 → 隐藏层2 64。每行是某个隐藏层2神经元对 128 个隐藏层1神经元的权重。' },
  W3: { W: () => nn.W3, rows: 10,  cols: 64,  grid: 5,  fanInIsImage: false, desc: 'W3: 隐藏层2 64 → 输出层 10。每行对应一个数字 (0-9) 的输出神经元，展示其对 64 个隐藏层2神经元的权重。' },
};

function renderWeights() {
  const cfg = WEIGHT_LAYERS[currentWeightLayer];
  const W = cfg.W();
  const nNeurons = cfg.rows;       // 输出神经元数 (= 矩阵行数)
  const fanIn = cfg.cols;          // 每个神经元的输入数 (= 矩阵列数)

  const container = weightCanvas.parentElement;
  const cw = container.clientWidth - 32;

  // W1 的输入是 28×28 图像，按图像比例显示；其余层把 fanIn 个权重排成方形小图
  const imgSide = cfg.fanInIsImage ? 28 : Math.ceil(Math.sqrt(fanIn));
  const gridCols = cfg.grid;
  const cellSize = Math.floor(cw / gridCols);
  const pixelSize = Math.max(2, Math.floor((cellSize - 2) / imgSide));
  const gap = 1;
  const thumbSize = pixelSize * imgSide + gap;

  const gridRows = Math.ceil(nNeurons / gridCols);
  const canvasW = gridCols * thumbSize;
  const canvasH = gridRows * thumbSize;

  weightCanvas.width = canvasW;
  weightCanvas.height = canvasH;

  wctx.fillStyle = '#1a1a2e';
  wctx.fillRect(0, 0, canvasW, canvasH);

  for (let n = 0; n < nNeurons; n++) {
    // 每个神经元独立归一化，确保细节始终可见
    let nMax = 0;
    for (let j = 0; j < fanIn; j++) nMax = Math.max(nMax, Math.abs(W[n][j]));
    if (nMax < 1e-8) nMax = 1;

    const col = n % gridCols;
    const row = Math.floor(n / gridCols);
    const bx = col * thumbSize;
    const by = row * thumbSize;

    for (let py = 0; py < imgSide; py++) {
      for (let px = 0; px < imgSide; px++) {
        const idx = py * imgSide + px;
        if (idx >= fanIn) continue;  // 非 W1 层 fanIn 可能不是完全平方数，余下留空
        const w = W[n][idx] / nMax;
        wctx.fillStyle = w < 0
          ? lerpColor('#ffffff', '#1565c0', -w)
          : lerpColor('#ffffff', '#ff6f00', w);
        wctx.fillRect(bx + px * pixelSize, by + py * pixelSize, pixelSize, pixelSize);
      }
    }

    wctx.strokeStyle = '#2a3f6f';
    wctx.lineWidth = 0.5;
    wctx.strokeRect(bx, by, thumbSize - gap, thumbSize - gap);

    // W3 输出层标注对应数字
    if (currentWeightLayer === 'W3') {
      wctx.fillStyle = '#e0e0e0';
      wctx.font = 'bold 10px sans-serif';
      wctx.textAlign = 'left';
      wctx.fillText(String(n), bx + 2, by + 11);
    }
  }

  // 更新底部说明
  const caption = document.getElementById('weightCaption');
  if (caption) caption.textContent = cfg.desc;
}

// 权重层切换标签
document.getElementById('weightTabs').addEventListener('click', (e) => {
  const btn = e.target.closest('.weight-tab');
  if (!btn) return;
  currentWeightLayer = btn.dataset.layer;
  document.querySelectorAll('.weight-tab').forEach(b => b.classList.toggle('active', b === btn));
  renderWeights();
});

function getActivations(data) {
  if (!data) return null;
  try {
    // 前向传播获取所有中间层激活值
    // 如果提供了 output 但不是完整的 cache，则重新前向
    const { cache, output } = nn.forward(data.input);
    return {
      input: data.input,
      a1: cache.a1,
      a2: cache.a2,
      output: data.output || output,
    };
  } catch (e) {
    return null;
  }
}

function drawNeuronDot(ctx, x, y, radius, value, isInput, isOutput) {
  // 输入层用灰阶
  if (isInput) {
    const intensity = Math.round(255 * (1 - value));
    ctx.fillStyle = `rgb(${intensity},${intensity},${intensity})`;
  } else if (isOutput) {
    // 输出层用专用色阶，微小的概率变化也能看到
    ctx.fillStyle = outputColor(value);
  } else {
    // 激活层用颜色映射
    ctx.fillStyle = activationColor(value);
  }
  ctx.beginPath();
  ctx.arc(x, y, Math.max(radius, 1), 0, Math.PI * 2);
  ctx.fill();
}

function activationColor(value) {
  // 将激活值映射到颜色: 低→蓝色, 中→灰色, 高→橙色
  if (value <= 0) {
    return '#1a237e'; // 深蓝 (无激活)
  } else if (value < 0.3) {
    // 蓝→灰
    const t = value / 0.3;
    return lerpColor('#1a237e', '#546e7a', t);
  } else if (value < 0.6) {
    // 灰→橙
    const t = (value - 0.3) / 0.3;
    return lerpColor('#546e7a', '#ff6f00', t);
  } else {
    // 高激活: 亮橙/红
    const t = Math.min((value - 0.6) / 0.4, 1);
    return lerpColor('#ff6f00', '#ff1744', t);
  }
}

function outputColor(value) {
  // 输出层专用色阶: 即使概率只有微小差异也能肉眼分辨
  // 50%→白绿, 30%→亮黄, 15%→橙蓝, 10%→紫, 5%以下→深紫
  if (value < 0.03) return '#0d001a';
  if (value < 0.08) return lerpColor('#0d001a', '#4a0072', (value - 0.03) / 0.05);
  if (value < 0.12) return lerpColor('#4a0072', '#004d40', (value - 0.08) / 0.04);
  if (value < 0.18) return lerpColor('#004d40', '#006064', (value - 0.12) / 0.06);
  if (value < 0.30) return lerpColor('#006064', '#00c853', (value - 0.18) / 0.12);
  if (value < 0.50) return lerpColor('#00c853', '#76ff03', (value - 0.30) / 0.20);
  if (value < 0.75) return lerpColor('#76ff03', '#ffea00', (value - 0.50) / 0.25);
  return lerpColor('#ffea00', '#ff1744', Math.min((value - 0.75) / 0.25, 1));
}

function lerpColor(c1, c2, t) {
  const r1 = parseInt(c1.slice(1, 3), 16);
  const g1 = parseInt(c1.slice(3, 5), 16);
  const b1 = parseInt(c1.slice(5, 7), 16);
  const r2 = parseInt(c2.slice(1, 3), 16);
  const g2 = parseInt(c2.slice(3, 5), 16);
  const b2 = parseInt(c2.slice(5, 7), 16);
  const r = Math.round(r1 + (r2 - r1) * t);
  const g = Math.round(g1 + (g2 - g1) * t);
  const b = Math.round(b1 + (b2 - b1) * t);
  return `rgb(${r},${g},${b})`;
}

// ── 导出和加载 ──────────────────────────────────────

document.getElementById('btnExport').addEventListener('click', () => {
  const data = nn.toJSON();
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `mnist_model_${nn.trainingCount}samples.json`;
  a.click();
  URL.revokeObjectURL(url);
});

document.getElementById('btnLoad').addEventListener('click', () => {
  document.getElementById('fileInput').click();
});

document.getElementById('fileInput').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (event) => {
    try {
      const data = JSON.parse(event.target.result);
      nn = NeuralNetwork.fromJSON(data);
      document.getElementById('statusSamples').textContent = nn.trainingCount;
      document.getElementById('statusLoss').textContent = '-';
      document.getElementById('statusNetwork').textContent = `已加载 (${nn.trainingCount} 样本)`;
      document.getElementById('lrSlider').value = nn.lr;
      document.getElementById('lrDisplay').textContent = nn.lr.toFixed(3);
      renderVisualization(null);
    } catch (err) {
      alert('加载模型失败: ' + err.message);
    }
  };
  reader.readAsText(file);
  // 重置 input 以便重复加载同一文件
  e.target.value = '';
});

// ── 清空事件 ────────────────────────────────────────

document.getElementById('btnClear').addEventListener('click', clearCanvas);

// ── 重置网络 ─────────────────────────────────────────

document.getElementById('btnRandomize').addEventListener('click', () => {
  if (confirm('确定要重新初始化网络吗？所有训练数据将丢失。')) {
    nn.randomize();
    document.getElementById('statusSamples').textContent = '0';
    document.getElementById('statusLoss').textContent = '-';
    document.getElementById('statusNetwork').textContent = '初始状态';
    document.getElementById('trainResult').classList.add('hidden');
    renderVisualization(null);
    initProbBars();
  }
});

// ── 学习率 ───────────────────────────────────────────

document.getElementById('lrSlider').addEventListener('input', (e) => {
  const lr = parseFloat(e.target.value);
  nn.lr = lr;
  document.getElementById('lrDisplay').textContent = lr.toFixed(3);
});

// ── 批量训练 ─────────────────────────────────────────

let batchTraining = false;
let mnistLoader = null;

// 页面加载时初始化 MNIST 数据
(async function initMNIST() {
  try {
    mnistLoader = new MNISTLoader();
    await mnistLoader.load('mnist_subset.json');
    console.log('MNIST 数据加载完成');
    document.getElementById('statusNetwork').textContent = `MNIST 已加载 (${mnistLoader.dataset.length} 样本)`;
  } catch (e) {
    console.warn('MNIST 数据加载失败，请确认 mnist_subset.json 存在:', e.message);
    document.getElementById('btnBatchTrain').disabled = true;
    document.getElementById('btnBatchTrain').textContent = '❌ 数据未加载';
  }
})();

document.getElementById('btnBatchTrain').addEventListener('click', async () => {
  if (batchTraining || !mnistLoader) return;

  const countPerDigit = parseInt(document.getElementById('batchCount').value);
  const btn = document.getElementById('btnBatchTrain');

  batchTraining = true;
  btn.disabled = true;
  btn.textContent = `⏳ 准备数据...`;
  document.getElementById('statusNetwork').textContent = '批量训练中...';

  // 从 MNIST 取类别平衡的数据
  const dataset = mnistLoader.getBalanced(countPerDigit);
  const total = dataset.length;
  shuffleArray(dataset);

  // 训练配置
  const epochs = Math.max(30, Math.round(200 / Math.sqrt(countPerDigit)));
  const effectiveTotal = total * epochs;
  const batchSize = 50;
  let trained = 0;
  let totalLoss = 0;

  btn.textContent = `⏳ 训练中 0/${effectiveTotal}...`;

  for (let epoch = 0; epoch < epochs; epoch++) {
    shuffleArray(dataset);

    let epochCorrect = 0;
    let epochLoss = 0;

    for (let i = 0; i < dataset.length; i += batchSize) {
      const batch = dataset.slice(i, i + batchSize);
      const result = nn.trainBatch(batch);
      totalLoss += result.loss;
      epochLoss += result.loss;
      trained += batch.length;
      epochCorrect += Math.round(result.accuracy * batch.length);

      const avgLoss = totalLoss / trained;
      const epochAcc = (epochCorrect / dataset.length * 100).toFixed(1);
      btn.textContent = `⏳ ${epoch + 1}/${epochs} acc:${epochAcc}% loss:${avgLoss.toFixed(3)}`;
      document.getElementById('statusLoss').textContent = avgLoss.toFixed(4);
      document.getElementById('statusSamples').textContent = nn.trainingCount;

      // 用当前批次的第一个样本刷新可视化
      renderVisualization({ input: batch[0].input });

      await sleep(0);
    }

    const epochAcc = (epochCorrect / dataset.length * 100).toFixed(1);
    document.getElementById('statusNetwork').textContent =
      `训练中 - Epoch ${epoch + 1}/${epochs} 准确率: ${epochAcc}%`;
  }

  // 训练完成 - 计算最终准确率
  batchTraining = false;
  btn.disabled = false;
  btn.textContent = '🚀 批量训练';

  // 在全部数据上评估准确率
  let correct = 0;
  for (const sample of dataset) {
    const { predictedClass } = nn.predict(sample.input);
    if (predictedClass === sample.label) correct++;
  }
  const accuracy = (correct / dataset.length * 100).toFixed(1);

  const finalLoss = totalLoss / trained;
  document.getElementById('statusNetwork').textContent =
    `已训练 ${nn.trainingCount} 样本 | 训练准确率: ${accuracy}%`;
  document.getElementById('statusLoss').textContent = finalLoss.toFixed(4);

  // 用第一个样本做可视化
  if (dataset.length > 0) {
    const { output } = nn.predict(dataset[0].input);
    renderVisualization({ input: dataset[0].input, output });
  }

  // 显示完成信息
  const trainResult = document.getElementById('trainResult');
  trainResult.classList.remove('hidden');
  trainResult.className = `train-result ${finalLoss > 1.0 ? 'loss-high' : 'loss-low'}`;
  trainResult.textContent = `✅ 批量训练完成! ${total} 样本 (每数字 ${countPerDigit} 个), 平均损失: ${finalLoss.toFixed(4)}`;

  // 切换识别模式预览效果
  setTimeout(() => {
    setMode('recognize');
  }, 500);
});

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function shuffleArray(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

// 禁用批量训练按钮当正在训练时
function updateBatchButton() {
  const btn = document.getElementById('btnBatchTrain');
  if (batchTraining) {
    btn.disabled = true;
  }
}

// ── 工具函数 ─────────────────────────────────────────

function argmax(arr) {
  let idx = 0;
  let max = arr[0];
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > max) { max = arr[i]; idx = i; }
  }
  return idx;
}

// ── 窗口大小变化 ─────────────────────────────────────

window.addEventListener('resize', () => {
  if (state.mode === 'recognize' && state.hasInk) {
    const input = sampleCanvas();
    const { output } = nn.predict(input);
    renderVisualization({ input, output });
  } else if (state.hasInk) {
    renderVisualization(null);
  } else {
    renderVisualization(null);
  }
});

// ── 键盘快捷键 ───────────────────────────────────────

document.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && state.mode === 'train') {
    const btn = document.getElementById('btnTrain');
    if (!btn.disabled) btn.click();
  }
  if (e.key === 'c' || e.key === 'C') {
    clearCanvas();
  }
  if ((e.key === 'r' || e.key === 'R') && state.mode === 'recognize' && state.hasInk) {
    doRecognize();
  }
  // 数字键 0-9 快速选择
  if (state.mode === 'train' && /^[0-9]$/.test(e.key)) {
    const checkboxes = document.querySelectorAll('#digitCheckboxes input[type="checkbox"]');
    clearCheckboxes();
    checkboxes[parseInt(e.key)].checked = true;
    updateTrainButton();
  }
});

// ── 说明页 ───────────────────────────────────────────

const helpModal = document.getElementById('helpModal');
document.getElementById('btnHelp').addEventListener('click', () => {
  helpModal.classList.remove('hidden');
});
document.getElementById('btnHelpClose').addEventListener('click', () => {
  helpModal.classList.add('hidden');
});
// 点击遮罩区域关闭
helpModal.addEventListener('click', (e) => {
  if (e.target === helpModal) helpModal.classList.add('hidden');
});
// ESC 关闭
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !helpModal.classList.contains('hidden')) {
    helpModal.classList.add('hidden');
  }
});

// ── 初始化 ──────────────────────────────────────────
initCheckboxes();
initProbBars();
renderVisualization(null);