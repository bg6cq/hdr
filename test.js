/**
 * test.js - 神经网络诊断测试
 *
 * 在浏览器控制台运行: testDiagnostics()
 * 或在页面底部查看诊断按钮
 */

function testDiagnostics() {
  console.log('%c╔════════════════════════════════════════╗', 'color: #00d2ff');
  console.log('%c║     神经网络诊断测试                   ║', 'color: #00d2ff');
  console.log('%c╚════════════════════════════════════════╝', 'color: #00d2ff');

  // ── 测试1: 简单模式识别 ──────────────────────────
  console.log('\n%c[测试1] 简单二分类: 全黑 vs 全白', 'color: #ffab00; font-weight: bold');

  const testNN = new NeuralNetwork(0.1);
  testNN.momentum = 0;  // 纯 SGD，排除动量影响

  const classA = new Float64Array(784); // 全0
  const classB = new Float64Array(784); // 全1
  for (let i = 0; i < 784; i++) {
    classB[i] = 1.0;
  }

  console.log(`  classA[0..5]: ${Array.from(classA.slice(0, 5)).map(v => v.toFixed(2)).join(', ')}`);
  console.log(`  classB[0..5]: ${Array.from(classB.slice(0, 5)).map(v => v.toFixed(2)).join(', ')}`);

  for (let epoch = 0; epoch < 50; epoch++) {
    let lossA = testNN.train(classA, 0).loss;
    let lossB = testNN.train(classB, 1).loss;

    if (epoch < 5 || epoch % 10 === 9) {
      const predA = testNN.predict(classA);
      const predB = testNN.predict(classB);
      console.log(`  Epoch ${epoch + 1}: lossA=${lossA.toFixed(4)} lossB=${lossB.toFixed(4)} | ` +
        `predA→${predA.predictedClass}(${(predA.output[0]*100).toFixed(1)}%) ` +
        `predB→${predB.predictedClass}(${(predB.output[1]*100).toFixed(1)}%)`);
    }
  }

  const finalA = testNN.predict(classA);
  const finalB = testNN.predict(classB);
  const passed1 = finalA.predictedClass === 0 && finalB.predictedClass === 1;
  console.log(`  结果: A→${finalA.predictedClass} B→${finalB.predictedClass} ${passed1 ? '✅ 通过' : '❌ 失败'}`);

  // ── 测试2: 检查反向传播符号 ──────────────────────
  console.log('\n%c[测试2] 梯度方向检查', 'color: #ffab00; font-weight: bold');

  const checkNN = new NeuralNetwork(0.1);
  checkNN.momentum = 0;

  // 用随机输入测试
  const testInput = new Float64Array(784);
  for (let i = 0; i < 784; i++) testInput[i] = Math.random() * 0.3;

  // 训练前
  const before = checkNN.predict(testInput);
  const beforeProb = before.output[3]; // 数字3的概率

  // 训练一次(label=3)
  checkNN.train(testInput, 3);

  // 训练后
  const after = checkNN.predict(testInput);
  const afterProb = after.output[3];

  console.log(`  输入: 随机像素, label=3`);
  console.log(`  训练前 P(3)=${(beforeProb*100).toFixed(2)}%`);
  console.log(`  训练后 P(3)=${(afterProb*100).toFixed(2)}%`);
  const passed2 = afterProb > beforeProb;
  console.log(`  结果: 概率${passed2 ? '上升 ✅' : '下降 ❌ 反向传播方向错误!'}`);

  // ── 测试3: 数据生成器质量 ─────────────────────────
  console.log('\n%c[测试3] 标准数据生成器质量', 'color: #ffab00; font-weight: bold');

  const gen = new DigitDataGenerator();
  const samples = gen.generate(3, { maxShift: 2, maxSlant: 0.2, minThickness: 2, maxThickness: 3, noiseLevel: 0.01 });

  for (let d = 0; d < 10; d++) {
    const s = samples[d * 3];
    const maxVal = Math.max(...s.input);
    const inkPixels = Array.from(s.input).filter(v => v > 0.2).length;
    const inkPercent = (inkPixels / 784 * 100).toFixed(1);
    console.log(`  数字 ${d}: 最大像素=${maxVal.toFixed(3)}, 有墨像素=${inkPixels}/${inkPixels}px (${inkPercent}%)`);
    console.log(DigitDataGenerator.visualizeSample(s.input, d));
  }

  // ── 测试4: 标准数据训练效果 ─────────────────────
  console.log('\n%c[测试4] 标准数据训练 (每数字5样本, 30 epoch)', 'color: #ffab00; font-weight: bold');

  const trainNN = new NeuralNetwork(0.1);
  trainNN.momentum = 0.0;  // 多分类用纯 SGD
  const dataset = gen.generate(5, { maxShift: 2, maxSlant: 0.2, minThickness: 2, maxThickness: 2.5, noiseLevel: 0.02 });

  shuffleArray(dataset);
  const totalSamples = dataset.length;

  for (let epoch = 0; epoch < 30; epoch++) {
    shuffleArray(dataset);
    let epochLoss = 0;
    let epochCorrect = 0;

    for (const sample of dataset) {
      const { output, loss } = trainNN.train(sample.input, sample.label);
      epochLoss += loss;
      if (argmax(output) === sample.label) epochCorrect++;
    }

    // 评估
    if (epoch < 5 || epoch % 5 === 4) {
      let correct = 0;
      for (const sample of dataset) {
        const { predictedClass } = trainNN.predict(sample.input);
        if (predictedClass === sample.label) correct++;
      }
      const acc = (correct / totalSamples * 100).toFixed(1);
      const avgLoss = (epochLoss / totalSamples).toFixed(4);
      console.log(`  Epoch ${epoch + 1}: 平均损失=${avgLoss}, 训练准确率=${acc}%`);
    }
  }

  // 最终评估
  let finalCorrect = 0;
  for (const sample of dataset) {
    const { predictedClass } = trainNN.predict(sample.input);
    if (predictedClass === sample.label) finalCorrect++;
  }
  console.log(`  最终准确率: ${(finalCorrect / totalSamples * 100).toFixed(1)}%`);
  console.log(`  结果: ${finalCorrect > totalSamples * 0.5 ? '✅ 通过 (>50%)' : finalCorrect > totalSamples * 0.2 ? '⚠️ 部分学习 (20-50%)' : '❌ 失败 (<20%)'}`);
}

// 在页面加载完后自动添加诊断按钮
if (typeof window !== 'undefined') {
  window.addEventListener('DOMContentLoaded', () => {
    // 在状态栏添加诊断按钮
    const statusBar = document.querySelector('.status-bar');
    if (statusBar) {
      const diagItem = document.createElement('div');
      diagItem.className = 'status-item';
      diagItem.innerHTML = `<button id="btnDiagnose" class="btn" style="padding:2px 8px;font-size:11px;">🔍 诊断</button>`;
      statusBar.appendChild(diagItem);

      document.getElementById('btnDiagnose').addEventListener('click', () => {
        console.clear();
        testDiagnostics();
        alert('诊断结果已输出到浏览器控制台 (按 F12 或 Cmd+Option+I 查看)');
      });
    }
  });
}

function shuffleArray(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

// 自执行检查
if (typeof window !== 'undefined') {
  // 加载完成时自动运行快速检查
  window.addEventListener('load', () => {
    console.log('ℹ️ 点击状态栏 "🔍 诊断" 按钮运行完整诊断测试');
  });
}