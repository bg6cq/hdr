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

  return input;
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

  if (mode === 'recognize' && state.hasInk) {
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
  if (!state.hasInk) return;

  const input = sampleCanvas();
  const { output, predictedClass } = nn.predict(input);

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

function setupVisCanvas() {
  const container = visCanvas.parentElement;
  const width = Math.min(container.clientWidth - 32, 600);
  const height = Math.max(container.clientHeight - 40, 450);
  visCanvas.width = width;
  visCanvas.height = height;
}

function renderVisualization(data) {
  setupVisCanvas();
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
          // 标签 (左侧，稍微左移)
          vctx.fillStyle = value > 0.5 ? '#00e676' : '#8899aa';
          vctx.font = value > 0.5 ? 'bold 11px sans-serif' : '10px sans-serif';
          vctx.textAlign = 'right';
          vctx.fillText(String(i), x - dotSize - 4, yPos + 4);

          // 概率值 (右侧)
          vctx.fillStyle = value > 0.1 ? '#e0e0e0' : '#555';
          vctx.font = '9px monospace';
          vctx.textAlign = 'left';
          vctx.fillText(`${(value * 100).toFixed(0)}%`, x + dotSize + 4, yPos + 4);
        }

        drawNeuronDot(vctx, x, yPos, dotSize, value, false);
      }
    }
  }

  // 层间连接线（装饰性）
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

function drawNeuronDot(ctx, x, y, radius, value, isInput) {
  // 输入层用灰阶
  if (isInput) {
    const intensity = Math.round(255 * (1 - value));
    ctx.fillStyle = `rgb(${intensity},${intensity},${intensity})`;
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

document.getElementById('btnBatchTrain').addEventListener('click', async () => {
  if (batchTraining) return;

  const countPerDigit = parseInt(document.getElementById('batchCount').value);
  const total = countPerDigit * 10;
  const btn = document.getElementById('btnBatchTrain');
  const origText = btn.textContent;

  batchTraining = true;
  btn.disabled = true;
  btn.textContent = `⏳ 训练中 0/${total}...`;
  document.getElementById('statusNetwork').textContent = '批量训练中...';

  // 生成标准数据 - 每个样本端点随机抖动，形状各不相同
  const gen = new DigitDataGenerator();
  const dataset = gen.generate(countPerDigit, {
    maxShift: 3,
    maxSlant: 0.3,
    minThickness: 1.0,
    maxThickness: 2.5,
    noiseLevel: 0.02,
    jitter: 1.8,        // 端点抖动: 每次生成的"3"笔画路径都不同
  });

  // 打乱数据
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
    // 只重绘可视化，不重新识别
    const input = sampleCanvas();
    const { output } = nn.predict(input);
    renderVisualization({ input, output });
  } else if (state.hasInk) {
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

// ── 初始化 ──────────────────────────────────────────

initCheckboxes();
initProbBars();
renderVisualization(null);