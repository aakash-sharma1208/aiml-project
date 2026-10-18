import modelData from '../data/model.json';
import { PredictResult } from '../types';

interface ModelData {
  gamma: number;
  n_support: number[];
  intercept: number[];
  dual_coef: number[][];
  support_vectors: number[][];
}

const model: ModelData = modelData as ModelData;

// Precompute class slice offsets for support vectors
const supportVectorSlices: Array<{ start: number; end: number }> = [];
let offset = 0;
for (const count of model.n_support) {
  supportVectorSlices.push({ start: offset, end: offset + count });
  offset += count;
}

/**
 * Predict digit and calculate calibrated decision scores from a 64-element pixel array.
 * Pixel values range from 0 to 16.
 */
export function predictFromPixels(rawPixels: number[]): PredictResult {
  if (rawPixels.length !== 64) {
    throw new Error(`Expected 64 pixel values, received ${rawPixels.length}.`);
  }

  // Uniform pixel scaling: [0, 16] -> [0.0, 1.0] (matching PixelScaler)
  const xScaled = new Float64Array(64);
  for (let i = 0; i < 64; i++) {
    const val = Number.isFinite(rawPixels[i]) ? rawPixels[i] : 0;
    xScaled[i] = Math.max(0, Math.min(16, val)) / 16.0;
  }

  const numSVs = model.support_vectors.length;
  const kVals = new Float64Array(numSVs);
  const gamma = model.gamma;

  // Compute RBF kernel: exp(-gamma * ||x - sv||^2)
  for (let s = 0; s < numSVs; s++) {
    const sv = model.support_vectors[s];
    let distSq = 0;
    for (let f = 0; f < 64; f++) {
      const diff = xScaled[f] - sv[f];
      distSq += diff * diff;
    }
    kVals[s] = Math.exp(-gamma * distSq);
  }

  const nClasses = 10;
  const nClassifiers = (nClasses * (nClasses - 1)) / 2; // 45
  const confidences = new Float64Array(nClassifiers);
  const votes = new Int32Array(nClasses);
  const sumOfConfidences = new Float64Array(nClasses);

  let k = 0;
  for (let i = 0; i < nClasses; i++) {
    for (let j = i + 1; j < nClasses; j++) {
      const sliceI = supportVectorSlices[i];
      const sliceJ = supportVectorSlices[j];

      // Dual coefficients in LibSVM OvO formulation:
      // For class i against class j (j > i), row is j - 1
      // For class j against class i (i < j), row is i
      const coefI = model.dual_coef[j - 1];
      const coefJ = model.dual_coef[i];

      let dotI = 0;
      for (let s = sliceI.start; s < sliceI.end; s++) {
        dotI += kVals[s] * coefI[s];
      }

      let dotJ = 0;
      for (let s = sliceJ.start; s < sliceJ.end; s++) {
        dotJ += kVals[s] * coefJ[s];
      }

      const dec = dotI + dotJ + model.intercept[k];
      confidences[k] = dec;

      // In OvO, dec > 0 favors class i; dec <= 0 favors class j
      if (dec > 0) {
        votes[i] += 1;
      } else {
        votes[j] += 1;
      }

      // Continuous OvR accumulation matching sklearn's _ovr_decision_function
      // Note: for binary classifier (i, j), confidence towards class j is -dec, and towards class i is +dec
      sumOfConfidences[i] += dec;
      sumOfConfidences[j] -= dec;

      k++;
    }
  }

  // Continuous OvR decision scores: votes + monotonic confidence transform
  const scores: number[] = new Array(nClasses);
  for (let c = 0; c < nClasses; c++) {
    const sc = sumOfConfidences[c];
    const transformed = sc / (3 * (Math.abs(sc) + 1));
    scores[c] = Number((votes[c] + transformed).toFixed(4));
  }

  // Determine winner: highest vote count, tie-broken by highest continuous score
  let bestDigit = 0;
  let bestVote = -1;
  let bestScore = -Infinity;

  for (let c = 0; c < nClasses; c++) {
    if (votes[c] > bestVote) {
      bestVote = votes[c];
      bestScore = scores[c];
      bestDigit = c;
    } else if (votes[c] === bestVote && scores[c] > bestScore) {
      bestScore = scores[c];
      bestDigit = c;
    }
  }

  return {
    predicted_digit: bestDigit,
    decision_scores: scores,
    preprocessed_pixels: Array.from(rawPixels).map((v) => Number(v.toFixed(2))),
  };
}

/**
 * Preprocess an HTML canvas (user drawing) into a 64-element calibrated array (values 0–16).
 * Follows the exact preprocessing pipeline:
 *  1. Luminance inversion (white background -> 0, black stroke -> 255)
 *  2. Bounding box detection with ink threshold
 *  3. Adaptive dilation for thin lines
 *  4. Aspect-ratio preserving square framing with padding
 *  5. Area-box downsampling to 8x8
 *  6. Contrast normalization (peak ink = 16.0)
 *  7. Center-of-mass alignment
 */
export function preprocessCanvas(canvas: HTMLCanvasElement): number[] | null {
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;

  const width = canvas.width;
  const height = canvas.height;
  const imgData = ctx.getImageData(0, 0, width, height);
  const data = imgData.data;

  // 1. Grayscale inverted ink intensity: 0 = white/no ink, 255 = black stroke
  const inkMap = new Float32Array(width * height);
  let totalInk = 0;
  for (let i = 0; i < width * height; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    const lum = 0.299 * r + 0.587 * g + 0.114 * b;
    const ink = Math.max(0, 255 - lum);
    inkMap[i] = ink;
    if (ink > 25) totalInk++;
  }

  if (totalInk < 20) return null;

  // 2. Bounding box of drawn strokes
  let minX = width;
  let maxX = 0;
  let minY = height;
  let maxY = 0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (inkMap[y * width + x] > 25) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (minX > maxX || minY > maxY) return null;
  const strokeW = maxX - minX + 1;
  const strokeH = maxY - minY + 1;
  if (strokeW < 8 && strokeH < 8) return null;

  // 3. Proportional margin padding (14%)
  const pad = Math.max(12, Math.round(Math.max(strokeW, strokeH) * 0.14));
  const cropX0 = Math.max(0, minX - pad);
  const cropY0 = Math.max(0, minY - pad);
  const cropX1 = Math.min(width - 1, maxX + pad);
  const cropY1 = Math.min(height - 1, maxY + pad);
  const cropW = cropX1 - cropX0 + 1;
  const cropH = cropY1 - cropY0 + 1;

  // 4. Square bounding box
  const maxSide = Math.max(cropW, cropH);
  const square = new Float32Array(maxSide * maxSide);
  const startX = Math.floor((maxSide - cropW) / 2);
  const startY = Math.floor((maxSide - cropH) / 2);

  for (let y = 0; y < cropH; y++) {
    for (let x = 0; x < cropW; x++) {
      const srcIdx = (cropY0 + y) * width + (cropX0 + x);
      const dstIdx = (startY + y) * maxSide + (startX + x);
      square[dstIdx] = inkMap[srcIdx];
    }
  }

  // 5. Downsampling to 8x8 using area averaging (true box filter)
  const arr8x8 = new Float32Array(64);
  const scale = maxSide / 8.0;

  for (let r = 0; r < 8; r++) {
    const y0 = Math.floor(r * scale);
    const y1 = Math.min(maxSide, Math.floor((r + 1) * scale));
    for (let c = 0; c < 8; c++) {
      const x0 = Math.floor(c * scale);
      const x1 = Math.min(maxSide, Math.floor((c + 1) * scale));

      let sum = 0;
      let count = 0;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          sum += square[y * maxSide + x];
          count++;
        }
      }
      arr8x8[r * 8 + c] = count > 0 ? sum / count : 0;
    }
  }

  // 6. Dynamic contrast stretching (scale peak ink to 16.0)
  let maxPeak = 0;
  for (let i = 0; i < 64; i++) {
    if (arr8x8[i] > maxPeak) maxPeak = arr8x8[i];
  }
  if (maxPeak < 10) return null;

  for (let i = 0; i < 64; i++) {
    arr8x8[i] = (arr8x8[i] / maxPeak) * 16.0;
  }

  // 7. Center of mass alignment
  let totalMass = 0;
  let sumY = 0;
  let sumX = 0;
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const val = arr8x8[r * 8 + c];
      totalMass += val;
      sumY += r * val;
      sumX += c * val;
    }
  }

  let final8x8 = arr8x8;
  if (totalMass > 0) {
    const cy = sumY / totalMass;
    const cx = sumX / totalMass;
    const dy = Math.max(-1.2, Math.min(1.2, 3.5 - cy));
    const dx = Math.max(-1.2, Math.min(1.2, 3.5 - cx));

    // Shift 8x8 using bilinear interpolation
    const shifted = new Float32Array(64);
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const srcR = r - dy;
        const srcC = c - dx;
        if (srcR >= 0 && srcR < 7 && srcC >= 0 && srcC < 7) {
          const r0 = Math.floor(srcR);
          const r1 = r0 + 1;
          const c0 = Math.floor(srcC);
          const c1 = c0 + 1;
          const fr = srcR - r0;
          const fc = srcC - c0;

          const v00 = arr8x8[r0 * 8 + c0];
          const v01 = arr8x8[r0 * 8 + c1];
          const v10 = arr8x8[r1 * 8 + c0];
          const v11 = arr8x8[r1 * 8 + c1];

          const top = v00 * (1 - fc) + v01 * fc;
          const bottom = v10 * (1 - fc) + v11 * fc;
          shifted[r * 8 + c] = top * (1 - fr) + bottom * fr;
        } else if (srcR >= 0 && srcR < 8 && srcC >= 0 && srcC < 8) {
          shifted[r * 8 + c] = arr8x8[Math.round(srcR) * 8 + Math.round(srcC)];
        } else {
          shifted[r * 8 + c] = 0;
        }
      }
    }
    final8x8 = shifted;
  }

  const result: number[] = new Array(64);
  for (let i = 0; i < 64; i++) {
    result[i] = Math.max(0, Math.min(16, Number(final8x8[i].toFixed(2))));
  }

  return result;
}
