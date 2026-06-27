/**
 * data.js - 合成手写数字数据集生成器
 *
 * 生成模拟 MNIST 风格的手写数字图片（28×28 灰度），
 * 每个数字带位置偏移、粗细变化、倾斜等变体。
 * 数字通过笔画轮廓线段渲染，生成连续粗线条。
 */

class DigitDataGenerator {
  /**
   * @param {number} count - 每个数字生成多少个样本
   * @param {object} [opts]
   * @param {number} [opts.maxShift=3] - 最大像素偏移
   * @param {number} [opts.minThickness=1.2] - 最小笔画粗细
   * @param {number} [opts.maxThickness=2.5] - 最大笔画粗细
   * @param {number} [opts.maxSlant=0.3] - 最大倾斜 (弧度 ≈ 17°)
   * @param {number} [opts.scaleMin=0.7] - 最小缩放
   * @param {number} [opts.scaleMax=1.0] - 最大缩放
   * @param {number} [opts.noiseLevel=0.03] - 背景噪声
   */
  generate(count, opts = {}) {
    const {
      maxShift = 3,
      minThickness = 1.2,
      maxThickness = 2.5,
      maxSlant = 0.3,
      scaleMin = 0.7,
      scaleMax = 1.0,
      noiseLevel = 0.03,
      jitter = 1.5,          // 线段端点随机抖动幅度
    } = opts;

    const dataset = [];
    for (let label = 0; label < 10; label++) {
      for (let i = 0; i < count; i++) {
        const shiftX = (Math.random() - 0.5) * 2 * maxShift;
        const shiftY = (Math.random() - 0.5) * 2 * maxShift;
        const thickness = minThickness + Math.random() * (maxThickness - minThickness);
        const slant = (Math.random() - 0.5) * 2 * maxSlant;
        const scale = scaleMin + Math.random() * (scaleMax - scaleMin);

        const img = this.renderDigit(label, { shiftX, shiftY, thickness, slant, scale, noiseLevel, jitter });
        dataset.push({ input: img, label });
      }
    }
    return dataset;
  }

  // ── 线段渲染器 ──────────────────────────────────

  renderDigit(digit, { shiftX, shiftY, thickness, slant, scale, noiseLevel, jitter = 1.5 }) {
    const grid = new Float64Array(784);
    let segments = this.getSegments(digit);

    // 对每个线段端点添加随机抖动 (每个端点独立抖动，让笔画路径变化)
    if (jitter > 0) {
      segments = segments.map(([x1, y1, x2, y2]) => [
        x1 + (Math.random() - 0.5) * 2 * jitter,
        y1 + (Math.random() - 0.5) * 2 * jitter,
        x2 + (Math.random() - 0.5) * 2 * jitter,
        y2 + (Math.random() - 0.5) * 2 * jitter,
      ]);
    }

    // 缩放线段
    const scaled = segments.map(([x1, y1, x2, y2]) => [
      x1 * scale, y1 * scale,
      x2 * scale, y2 * scale,
    ]);

    const cx = 14 + shiftX;
    const cy = 14 + shiftY;

    // 对每个像素计算到最近线段距离
    for (let py = 0; py < 28; py++) {
      for (let px = 0; px < 28; px++) {
        // 倾斜变换
        const tx = (px - cx) - (py - cy) * Math.tan(slant);
        const ty = py - cy;

        // 计算到最近线段的最小距离
        let minDist = Infinity;
        for (const [x1, y1, x2, y2] of scaled) {
          const d = pointToSegmentDist(tx, ty, x1, y1, x2, y2);
          if (d < minDist) minDist = d;
        }

        const halfThick = thickness / 2;
        let val;
        if (minDist <= halfThick) {
          // 笔画内部: 平滑过渡
          val = 0.7 + 0.3 * (1 - minDist / halfThick);
        } else {
          // 笔画边缘羽化 + 背景噪声
          const falloff = Math.max(0, 1 - (minDist - halfThick) / 1.5);
          val = Math.random() * noiseLevel + 0.3 * falloff * (minDist <= halfThick + 1.5 ? 1 : 0);
        }

        grid[py * 28 + px] = Math.max(0, Math.min(1, val));
      }
    }

    return grid;
  }

  // ── 数字笔画线段模板 ────────────────────────────
  // 每个数字由若干连续笔画段组成 (x1,y1,x2,y2)
  // 坐标范围约 [-7, 7]，在 28x28 网格上居中渲染

  getSegments(digit) {
    const segs = {
      0: [
        [-4,-6, -2,-7], [-2,-7, 2,-7], [2,-7, 4,-6],
        [4,-6, 5,-4], [5,-4, 5,-2], [5,-2, 5,0],
        [5,0, 5,2], [5,2, 5,4], [5,4, 4,6],
        [4,6, 2,7], [2,7, -2,7], [-2,7, -4,6],
        [-4,6, -5,4], [-5,4, -5,2], [-5,2, -5,0],
        [-5,0, -5,-2], [-5,-2, -5,-4], [-5,-4, -4,-6],
      ],
      1: [
        [-2,-7, 2,-7],
        [0,-7, 0,-5], [0,-5, 0,-3], [0,-3, 0,-1],
        [0,-1, 0,1], [0,1, 0,3], [0,3, 0,5], [0,5, 0,7],
        // 底部横线装饰
        [-2,7, 2,7],
      ],
      2: [
        [-5,-6, -4,-7], [-4,-7, -2,-7], [-2,-7, 0,-7],
        [0,-7, 2,-7], [2,-7, 4,-7],
        [4,-7, 5,-6], [5,-6, 5,-4], [5,-4, 5,-2],
        [5,-2, 4,-1], [4,-1, 2,0],
        [2,0, 0,1], [0,1, -2,2], [-2,2, -4,3],
        [-4,3, -5,4], [-5,4, -5,6],
        [-5,6, -4,7], [-4,7, -2,7], [-2,7, 0,7],
        [0,7, 2,7], [2,7, 4,7], [4,7, 5,7],
      ],
      3: [
        [-4,-7, -2,-7], [-2,-7, 2,-7], [2,-7, 4,-7],
        [4,-7, 5,-5], [5,-5, 5,-3], [5,-3, 4,-1],
        [4,-1, 2,-1], [2,-1, -1,-1],
        [2,-1, 4,0], [4,0, 5,2],
        [5,2, 5,5], [5,5, 4,7], [4,7, 2,7],
        [2,7, -2,7], [-2,7, -4,7],
      ],
      4: [
        [-5,-7, -4,-5], [-4,-5, -3,-3], [-3,-3, -2,-1],
        [-2,-1, -1,1], [-1,1, 0,3],
        [0,3, 0,5], [0,5, 0,7],
        [5,-7, 5,-5], [5,-5, 5,-3], [5,-3, 5,-1],
        [5,-1, 5,1], [5,1, 5,3], [5,3, 5,5], [5,5, 5,7],
        [-1,1, 1,1], [1,1, 3,1], [3,1, 5,1],
      ],
      5: [
        [-4,-7, -2,-7], [-2,-7, 0,-7], [0,-7, 2,-7],
        [-5,-5, -5,-3], [-5,-3, -5,-1],
        [-5,-1, -4,0], [-4,0, -2,0],
        [-2,0, 0,0], [0,0, 2,0], [2,0, 4,0],
        [4,0, 5,2], [5,2, 5,4], [5,4, 5,6],
        [5,6, 4,7], [4,7, 2,7],
        [2,7, -2,7], [-2,7, -4,7], [-4,7, -5,6],
      ],
      6: [
        [-3,-7, -5,-5], [-5,-5, -5,-3], [-5,-3, -5,-1],
        [-5,-1, -5,1], [-5,1, -5,3], [-5,3, -5,5],
        [-5,5, -4,7], [-4,7, -2,7],
        [-2,7, 0,7], [0,7, 2,7], [2,7, 4,7],
        [4,7, 5,5], [5,5, 5,3], [5,3, 4,1],
        [4,1, 2,0], [2,0, 0,0],
        [0,0, -2,0], [-2,0, -3,-1],
        [0,-3, 2,-4], [2,-4, 3,-6],
      ],
      7: [
        [-5,-7, -3,-7], [-3,-7, -1,-7],
        [-1,-7, 1,-7], [1,-7, 3,-7], [3,-7, 5,-7],
        [4,-5, 3,-3], [3,-3, 2,-1],
        [2,-1, 1,1], [1,1, 0,3],
        [0,3, -1,5], [-1,5, -2,7],
      ],
      8: [
        [-3,-7, -1,-7], [-1,-7, 1,-7], [1,-7, 3,-7],
        [3,-7, 4,-5], [4,-5, 4,-3], [4,-3, 3,-1],
        [3,-1, 1,0], [1,0, -1,0],
        [-1,0, -3,-1], [-3,-1, -4,-3],
        [-4,-3, -4,-5], [-4,-5, -3,-7],
        [-4,1, -4,3], [-4,3, -4,5],
        [-4,5, -3,7], [-3,7, -1,7],
        [-1,7, 1,7], [1,7, 3,7],
        [3,7, 4,5], [4,5, 4,3], [4,3, 3,1],
        [3,1, 1,0],
      ],
      9: [
        [-3,-7, -1,-7], [-1,-7, 1,-7], [1,-7, 3,-7],
        [3,-7, 4,-5], [4,-5, 4,-3], [4,-3, 4,-1],
        [4,-1, 4,1], [4,1, 3,3],
        [3,3, 1,4], [1,4, -1,4],
        [-1,4, -3,3], [-3,3, -4,1],
        [-4,1, -4,-1], [-4,-1, -4,-3], [-4,-3, -3,-5],
        [1,6, 1,7], [0,7, 2,7],
      ],
    };
    return segs[digit] || [];
  }

  // ── 工具: 可视化某个样本 ────────────────────────

  static visualizeSample(input, label) {
    let s = `数字 ${label}:\n`;
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const v = input[y * 28 + x];
        s += v > 0.5 ? '██' : v > 0.2 ? '░░' : '  ';
      }
      s += '\n';
    }
    return s;
  }

  /** 性能测试 */
  static benchmark(count = 5) {
    const gen = new DigitDataGenerator();
    const start = performance.now();
    const data = gen.generate(count);
    const ms = performance.now() - start;
    console.log(`生成 ${data.length} 样本耗时 ${ms.toFixed(0)}ms (平均 ${(ms / data.length).toFixed(1)}ms/样本)`);
    // 显示每个数字的第一个样本
    for (let i = 0; i < 10; i++) {
      console.log(DigitDataGenerator.visualizeSample(data[i * count].input, i));
    }
    return data;
  }
}


// ═══════════════════════════════════════════════════════
//  几何工具函数
// ═══════════════════════════════════════════════════════

/** 计算点 P 到线段 AB 的最短距离 */
function pointToSegmentDist(px, py, ax, ay, bx, by) {
  const abx = bx - ax;
  const aby = by - ay;
  const apx = px - ax;
  const apy = py - ay;

  // 投影系数 t
  const dot = apx * abx + apy * aby;
  const lenSq = abx * abx + aby * aby;

  let t;
  if (lenSq === 0) {
    t = 0; // 线段退化为点
  } else {
    t = Math.max(0, Math.min(1, dot / lenSq));
  }

  // 最近点坐标
  const cx = ax + t * abx;
  const cy = ay + t * aby;

  const dx = px - cx;
  const dy = py - cy;
  return Math.sqrt(dx * dx + dy * dy);
}

// 导出到全局
window.DigitDataGenerator = DigitDataGenerator;