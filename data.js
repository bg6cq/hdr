/**
 * data.js - MNIST 数据集加载器
 *
 * 从 mnist_subset.json (每数字 500 样本, 共 5000) 加载真实手写数字数据。
 * 支持在浏览器中远程加载和本地预加载（数据嵌入）。
 *
 * 用法:
 *   const loader = new MNISTLoader();
 *   await loader.load('mnist_subset.json');
 *   const dataset = loader.getDataset();
 *   const batch = loader.getBatch(50);
 */

class MNISTLoader {
  constructor() {
    this.dataset = null;
    this.loaded = false;
  }

  /**
   * 从 JS 嵌入变量或 JSON 文件加载 MNIST 数据
   * @param {string} url - JSON 文件路径（仅在 window.MNIST_DATA 不存在时使用）
   */
  async load(url) {
    let raw;

    // 优先使用 <script> 嵌入的数据（解决 file:// 协议下 fetch 被拦截的问题）
    if (window.MNIST_DATA) {
      raw = window.MNIST_DATA;
      console.log('MNIST: 使用嵌入数据');
    } else {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`加载失败: ${resp.status}`);
      raw = await resp.json();
    }

    this.dataset = raw.map(item => ({
      input: new Float64Array(item.input.map(v => v / 255)),
      label: item.label,
    }));

    this.loaded = true;

    // 统计信息
    const counts = new Array(10).fill(0);
    for (const s of this.dataset) counts[s.label]++;

    console.log(`MNIST 已加载: ${this.dataset.length} 样本`);
    for (let d = 0; d < 10; d++) {
      console.log(`  数字 ${d}: ${counts[d]}`);
    }

    return this.dataset;
  }

  /** 获取全部数据集 */
  getDataset() {
    if (!this.dataset) throw new Error('数据未加载，请先调用 load()');
    return this.dataset;
  }

  /**
   * 获取按数字分组的副本 (训练用, 保证每批次的类别平衡)
   * @param {number} perDigit - 每数字取多少样本
   * @returns {Array} [{ input, label }]
   */
  getBalanced(perDigit = 50) {
    if (!this.dataset) throw new Error('数据未加载');

    // 按 label 分组
    const groups = Array.from({ length: 10 }, () => []);
    for (const s of this.dataset) {
      groups[s.label].push(s);
    }

    const result = [];
    for (let d = 0; d < 10; d++) {
      const n = Math.min(perDigit, groups[d].length);
      // 打乱后取前 n 个
      const shuffled = [...groups[d]];
      for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
      }
      for (let i = 0; i < n; i++) {
        result.push(shuffled[i]);
      }
    }

    return result;
  }

  /** 获取指定数量随机样本 */
  getRandom(count = 100) {
    if (!this.dataset) throw new Error('数据未加载');
    const shuffled = [...this.dataset];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled.slice(0, count);
  }

  /** 获取一批次 (按 label 采样保证类别平衡) */
  getBatch(batchSize = 50) {
    if (!this.dataset) throw new Error('数据未加载');
    return this.getBalanced(batchSize / 10);
  }

  /** 显示某个样本的 ASCII 可视化 */
  static visualize(input, label) {
    let s = `数字 ${label}:\n`;
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const v = input[y * 28 + x];
        s += v > 0.5 ? '██' : v > 0.15 ? '░░' : '  ';
      }
      s += '\n';
    }
    return s;
  }
}

// 导出到全局
window.MNISTLoader = MNISTLoader;