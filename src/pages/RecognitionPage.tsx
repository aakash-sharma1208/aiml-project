import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Chart, registerables } from 'chart.js';
import datasetData from '../data/dataset.json';
import { DatasetInfo, DatasetSample, PredictResult } from '../types';
import { preprocessCanvas, predictFromPixels } from '../utils/svm';
import { Sparkles, RotateCcw, Trash2, ArrowRight, Eye, AlertCircle } from 'lucide-react';

Chart.register(...registerables);

const data = datasetData as unknown as DatasetInfo;

type ResultState = 'placeholder' | 'loading' | 'content' | 'error';
type StrokeWidth = 16 | 22 | 28;

export const RecognitionPage: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const previewCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const chartCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const chartInstanceRef = useRef<Chart | null>(null);

  const [strokeWidth, setStrokeWidth] = useState<StrokeWidth>(22);
  const [resultState, setResultState] = useState<ResultState>('placeholder');
  const [errorMessage, setErrorMessage] = useState('');
  const [prediction, setPrediction] = useState<PredictResult | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const strokesRef = useRef<Array<Array<{ x: number; y: number }>>>([]);
  const currentStrokeRef = useRef<Array<{ x: number; y: number }> | null>(null);

  // Initialize canvas with pure white background
  const initCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = strokeWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#000000';
  }, [strokeWidth]);

  useEffect(() => {
    initCanvas();
  }, [initCanvas]);

  // Coordinate helper
  const getCoordinates = (e: React.MouseEvent | React.TouchEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    if ('touches' in e && e.touches.length > 0) {
      const t = e.touches[0];
      return {
        x: (t.clientX - rect.left) * scaleX,
        y: (t.clientY - rect.top) * scaleY,
      };
    }
    const mouseEvent = e as React.MouseEvent;
    return {
      x: (mouseEvent.clientX - rect.left) * scaleX,
      y: (mouseEvent.clientY - rect.top) * scaleY,
    };
  };

  const startDraw = (e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.lineWidth = strokeWidth;
    ctx.strokeStyle = '#000000';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    setIsDrawing(true);
    const pos = getCoordinates(e);
    currentStrokeRef.current = [pos];

    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
    ctx.arc(pos.x, pos.y, strokeWidth / 2, 0, Math.PI * 2);
    ctx.fillStyle = '#000000';
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
  };

  const draw = (e: React.MouseEvent | React.TouchEvent) => {
    if (!isDrawing) return;
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const pos = getCoordinates(e);
    if (currentStrokeRef.current) {
      currentStrokeRef.current.push(pos);
    }
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
  };

  const endDraw = () => {
    if (!isDrawing) return;
    setIsDrawing(false);
    if (currentStrokeRef.current && currentStrokeRef.current.length > 0) {
      strokesRef.current.push(currentStrokeRef.current);
    }
    currentStrokeRef.current = null;
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) ctx.beginPath();
    }
  };

  const handleClear = () => {
    strokesRef.current = [];
    currentStrokeRef.current = null;
    initCanvas();
    setResultState('placeholder');
    setPrediction(null);
    setIsScanning(false);
  };

  const handleUndo = () => {
    if (strokesRef.current.length === 0) return;
    strokesRef.current.pop();
    initCanvas();

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    strokesRef.current.forEach((stroke) => {
      if (stroke.length === 0) return;
      ctx.beginPath();
      ctx.moveTo(stroke[0].x, stroke[0].y);
      stroke.forEach((pt) => {
        ctx.lineTo(pt.x, pt.y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(pt.x, pt.y);
      });
      if (stroke.length === 1) {
        ctx.arc(stroke[0].x, stroke[0].y, strokeWidth / 2, 0, Math.PI * 2);
        ctx.fillStyle = '#000000';
        ctx.fill();
      }
      ctx.beginPath();
    });

    setResultState('placeholder');
    setPrediction(null);
  };

  // Render 8x8 preview canvas
  const renderPreviewCanvas = (pixels: number[]) => {
    const canvas = previewCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.imageSmoothingEnabled = false;
    const size = canvas.width;
    const cellSize = size / 8;
    ctx.fillStyle = '#060a12';
    ctx.fillRect(0, 0, size, size);

    for (let i = 0; i < 64; i++) {
      const row = Math.floor(i / 8);
      const col = i % 8;
      const val = pixels[i];
      const intensity = Math.round((Math.max(0, Math.min(16, val)) / 16) * 255);
      // Cool cyan-tinted pixel intensity
      ctx.fillStyle = `rgb(${Math.round(intensity * 0.7)}, ${Math.round(intensity * 0.95)}, ${intensity})`;
      ctx.fillRect(col * cellSize, row * cellSize, cellSize, cellSize);
    }
  };

  // Render scores bar chart with Chart.js
  const renderScoresChart = useCallback((scores: number[], predictedDigit: number) => {
    const canvas = chartCanvasRef.current;
    if (!canvas) return;

    if (chartInstanceRef.current) {
      chartInstanceRef.current.destroy();
      chartInstanceRef.current = null;
    }

    const labels = Array.from({ length: 10 }, (_, i) => `${i}`);
    const backgroundColors = scores.map((_, i) =>
      i === predictedDigit
        ? 'rgba(6, 182, 212, 0.9)'
        : 'rgba(56, 189, 248, 0.25)'
    );
    const borderColors = scores.map((_, i) =>
      i === predictedDigit ? 'rgba(34, 211, 238, 1)' : 'rgba(56, 189, 248, 0.5)'
    );

    chartInstanceRef.current = new Chart(canvas, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'Decision Score',
            data: scores,
            backgroundColor: backgroundColors,
            borderColor: borderColors,
            borderWidth: 1.5,
            borderRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
          duration: 600,
          easing: 'easeOutQuart',
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(12, 19, 34, 0.9)',
            titleColor: '#38bdf8',
            bodyColor: '#f8fafc',
            borderColor: 'rgba(56, 189, 248, 0.3)',
            borderWidth: 1,
            padding: 10,
            displayColors: false,
            callbacks: {
              title: (items) => `Class: Digit ${items[0].label}`,
              label: (item) => `Decision Score: ${item.formattedValue}`,
            },
          },
        },
        scales: {
          x: {
            ticks: {
              color: '#94a3b8',
              font: { size: 11, weight: 'bold' },
            },
            grid: { color: 'rgba(255,255,255,0.04)' },
          },
          y: {
            ticks: {
              color: '#94a3b8',
              font: { size: 10 },
            },
            grid: { color: 'rgba(255,255,255,0.04)' },
          },
        },
      },
    });
  }, []);

  // Update preview canvas and chart when prediction state changes
  useEffect(() => {
    if (resultState === 'content' && prediction) {
      renderPreviewCanvas(prediction.preprocessed_pixels);
      renderScoresChart(prediction.decision_scores, prediction.predicted_digit);
    }
  }, [resultState, prediction, renderScoresChart]);

  // Clean up chart on unmount
  useEffect(() => {
    return () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
        chartInstanceRef.current = null;
      }
    };
  }, []);

  const handlePredict = () => {
    if (strokesRef.current.length === 0) {
      setErrorMessage('Please draw a digit in the canvas before predicting.');
      setResultState('error');
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas) return;

    setResultState('loading');
    setIsScanning(true);

    setTimeout(() => {
      try {
        const pixels = preprocessCanvas(canvas);
        if (!pixels) {
          setIsScanning(false);
          setErrorMessage('The canvas appears to be blank. Please draw a digit clearly.');
          setResultState('error');
          return;
        }

        const result = predictFromPixels(pixels);
        setIsScanning(false);
        setPrediction(result);
        setResultState('content');
      } catch (err) {
        setIsScanning(false);
        setErrorMessage(err instanceof Error ? err.message : 'Prediction failed');
        setResultState('error');
      }
    }, 250);
  };

  const handleTestExample = (sample: DatasetSample) => {
    setResultState('loading');
    setIsScanning(false);

    setTimeout(() => {
      try {
        const result = predictFromPixels(sample.flat);
        setPrediction(result);
        setResultState('content');
      } catch (err) {
        setErrorMessage(err instanceof Error ? err.message : 'Failed to test example');
        setResultState('error');
      }
    }, 150);
  };

  // Pick one sample for each digit class (0-9)
  const exampleDigits: DatasetSample[] = [];
  for (let digit = 0; digit <= 9; digit++) {
    const match = data.samples.find((s) => s.label === digit);
    if (match) exampleDigits.push(match);
  }

  return (
    <section className="page" id="page-recognition">
      <div className="page-header">
        <span className="header-tag">
          <Sparkles size={14} /> Neural Inference Engine
        </span>
        <h1>Digit Recognition Workspace</h1>
        <p>
          Draw any handwritten digit (0–9) on the canvas to evaluate it against the Support Vector
          Machine classifier.
        </p>
      </div>

      <div className="recognition-grid">
        {/* Canvas panel */}
        <div className="card canvas-panel">
          <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h2 className="card-title">
              <span className="brand-icon" style={{ width: '28px', height: '28px', borderRadius: '6px' }}>
                <Sparkles size={15} />
              </span>
              Drawing Canvas
            </h2>
            <div className="stroke-selector">
              <button
                type="button"
                className={`stroke-btn ${strokeWidth === 16 ? 'active' : ''}`}
                onClick={() => setStrokeWidth(16)}
              >
                Fine
              </button>
              <button
                type="button"
                className={`stroke-btn ${strokeWidth === 22 ? 'active' : ''}`}
                onClick={() => setStrokeWidth(22)}
              >
                Normal
              </button>
              <button
                type="button"
                className={`stroke-btn ${strokeWidth === 28 ? 'active' : ''}`}
                onClick={() => setStrokeWidth(28)}
              >
                Bold
              </button>
            </div>
          </div>

          <div className={`canvas-frame-container ${isDrawing ? 'is-drawing' : ''}`}>
            <div className="canvas-frame">
              <canvas
                ref={canvasRef}
                id="digitCanvas"
                width={280}
                height={280}
                aria-label="Digit drawing canvas"
                onMouseDown={startDraw}
                onMouseMove={draw}
                onMouseUp={endDraw}
                onMouseLeave={endDraw}
                onTouchStart={startDraw}
                onTouchMove={draw}
                onTouchEnd={endDraw}
              />
              {isScanning && <div className="scan-beam" />}
            </div>

            <div className="canvas-toolbar">
              <div className="canvas-actions">
                <button
                  type="button"
                  className="btn btn-ghost"
                  onClick={handleUndo}
                  title="Undo stroke"
                >
                  <RotateCcw size={16} /> Undo
                </button>
                <button
                  type="button"
                  className="btn btn-ghost"
                  onClick={handleClear}
                  title="Clear canvas"
                >
                  <Trash2 size={16} /> Clear
                </button>
              </div>

              <button
                type="button"
                className="btn btn-primary"
                onClick={handlePredict}
                disabled={resultState === 'loading'}
              >
                <Sparkles size={16} /> Predict Digit
              </button>
            </div>
          </div>
        </div>

        {/* Result panel */}
        <div className="card result-panel">
          {resultState === 'placeholder' && (
            <div className="result-empty">
              <div className="result-empty-icon">
                <Eye size={36} />
              </div>
              <div>
                <h3 style={{ color: '#fff', fontSize: '1.1rem', fontWeight: 700, marginBottom: '0.4rem' }}>
                  Awaiting Input
                </h3>
                <p>
                  Draw a digit in the canvas and press <strong>Predict Digit</strong> or choose a
                  preset sample below to test.
                </p>
              </div>
            </div>
          )}

          {resultState === 'loading' && (
            <div className="state-center">
              <div className="cool-spinner" />
              <p style={{ color: 'var(--sky-bright)', fontWeight: 600 }}>
                Extracting 8×8 resolution matrix & calculating decision hyperplanes...
              </p>
            </div>
          )}

          {resultState === 'error' && (
            <div className="state-center">
              <AlertCircle size={44} color="var(--rose-alert)" />
              <p style={{ color: '#f8fafc', fontSize: '0.95rem' }}>{errorMessage}</p>
              <button
                type="button"
                className="btn btn-ghost"
                onClick={() => setResultState('placeholder')}
              >
                Try Again
              </button>
            </div>
          )}

          {resultState === 'content' && prediction && (
            <div className="result-success">
              <div className="predicted-hero">
                <div className="digit-pod">
                  <span className="digit-pod-num">{prediction.predicted_digit}</span>
                </div>
                <div className="predicted-details">
                  <span className="predicted-tag">SVM Prediction</span>
                  <div className="predicted-headline">
                    Identified as Digit {prediction.predicted_digit}
                  </div>
                  <div className="predicted-sub">
                    Optimal separating margin across 10 classes in high-dimensional RBF space.
                  </div>
                </div>
              </div>

              {/* Matrix + Chart Split */}
              <div className="eval-split">
                <div className="matrix-inspector-wrap">
                  <canvas
                    ref={previewCanvasRef}
                    width={96}
                    height={96}
                    className="preview-canvas-cool"
                    title="8×8 Calibrated Matrix Representation"
                  />
                  <span className="matrix-label">8×8 AI View</span>
                </div>

                <div className="scores-container">
                  <div className="scores-header">
                    <span className="scores-title">Decision Scores</span>
                    <span className="scores-note">Higher = Stronger confidence</span>
                  </div>
                  <div style={{ position: 'relative', height: '140px' }}>
                    <canvas ref={chartCanvasRef} />
                  </div>
                </div>
              </div>

              <button
                type="button"
                className="btn btn-ghost btn-full"
                onClick={handleClear}
              >
                Clear & Draw Another Digit
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Preset example digits */}
      <div className="card presets-shelf">
        <div className="presets-header">
          <div>
            <h2 className="card-title">
              <ArrowRight size={18} color="var(--cyan-glow)" /> Quick-Test Dataset Samples
            </h2>
            <p className="card-desc" style={{ marginBottom: 0 }}>
              Click any known dataset digit to test instant pipeline inference
            </p>
          </div>
        </div>

        <div className="presets-grid">
          {exampleDigits.map((sample) => (
            <div
              key={sample.label}
              className="preset-card"
              title={`Click to test Digit ${sample.label}`}
              onClick={() => handleTestExample(sample)}
            >
              <canvas
                width={52}
                height={52}
                className="preset-canvas"
                ref={(el) => {
                  if (el) {
                    const ctx = el.getContext('2d');
                    if (ctx) {
                      ctx.imageSmoothingEnabled = false;
                      const cellSize = 52 / 8;
                      ctx.fillStyle = '#060a12';
                      ctx.fillRect(0, 0, 52, 52);
                      for (let row = 0; row < 8; row++) {
                        for (let col = 0; col < 8; col++) {
                          const val = sample.image[row][col];
                          const intensity = Math.round((val / 16) * 255);
                          ctx.fillStyle = `rgb(${Math.round(intensity * 0.7)}, ${Math.round(intensity * 0.95)}, ${intensity})`;
                          ctx.fillRect(col * cellSize, row * cellSize, cellSize, cellSize);
                        }
                      }
                    }
                  }
                }}
              />
              <span className="preset-digit-tag">Digit {sample.label}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};
