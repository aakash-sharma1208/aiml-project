import React, { useState, useEffect, useRef } from 'react';
import { Chart, registerables } from 'chart.js';
import datasetData from '../data/dataset.json';
import { DatasetInfo, DatasetSample } from '../types';
import { Database, Image, BarChart3, Layers, Sparkles, CheckCircle2 } from 'lucide-react';

Chart.register(...registerables);

const data = datasetData as unknown as DatasetInfo;

export const DatasetPage: React.FC<{ onTestSample?: (sample: DatasetSample) => void }> = ({
  onTestSample,
}) => {
  const chartCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const chartInstanceRef = useRef<Chart | null>(null);

  const [selectedDigit, setSelectedDigit] = useState<number | 'all'>('all');
  const [activeSample, setActiveSample] = useState<DatasetSample>(data.samples[0]);

  const meta = data.metadata;
  const dist = data.class_distribution;
  const samples = data.samples;

  // Filter samples based on selection
  const filteredSamples =
    selectedDigit === 'all'
      ? samples
      : samples.filter((s) => s.label === selectedDigit);

  useEffect(() => {
    const canvas = chartCanvasRef.current;
    if (!canvas) return;

    if (chartInstanceRef.current) {
      chartInstanceRef.current.destroy();
      chartInstanceRef.current = null;
    }

    const labels = Object.keys(dist)
      .sort((a, b) => Number(a) - Number(b))
      .map((k) => `Digit ${k}`);
    const vals = labels.map((l) => dist[l.replace('Digit ', '')] ?? 0);

    chartInstanceRef.current = new Chart(canvas, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'Sample Count',
            data: vals,
            backgroundColor: 'rgba(6, 182, 212, 0.65)',
            borderColor: 'rgba(34, 211, 238, 1)',
            borderWidth: 1.5,
            borderRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
          duration: 700,
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
          },
        },
        scales: {
          x: {
            ticks: { color: '#94a3b8', font: { weight: 'bold' } },
            grid: { color: 'rgba(255,255,255,0.04)' },
          },
          y: {
            min: 150,
            ticks: { color: '#94a3b8' },
            grid: { color: 'rgba(255,255,255,0.04)' },
          },
        },
      },
    });

    return () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
        chartInstanceRef.current = null;
      }
    };
  }, [dist]);

  return (
    <section className="page" id="page-dataset">
      <div className="page-header">
        <span className="header-tag">
          <Database size={14} /> Dataset Explorer &amp; Pixel Matrix Inspector
        </span>
        <h1>Scikit-Learn Digits Repository</h1>
        <p>
          Explore 1,797 normalized 8×8 handwritten digits. Inspect individual pixel intensities and
          class distributions.
        </p>
      </div>

      {/* Summary stats */}
      <div className="dataset-summary-grid">
        <div className="stat-pod">
          <div className="stat-pod-icon cyan">
            <Database size={24} />
          </div>
          <div className="stat-pod-info">
            <span className="stat-pod-label">Total Images</span>
            <span className="stat-pod-value">{meta.n_samples?.toLocaleString()}</span>
          </div>
        </div>

        <div className="stat-pod">
          <div className="stat-pod-icon blue">
            <Image size={24} />
          </div>
          <div className="stat-pod-info">
            <span className="stat-pod-label">Resolution</span>
            <span className="stat-pod-value">8×8 Px</span>
          </div>
        </div>

        <div className="stat-pod">
          <div className="stat-pod-icon indigo">
            <Layers size={24} />
          </div>
          <div className="stat-pod-info">
            <span className="stat-pod-label">Pixel Depth</span>
            <span className="stat-pod-value">0–16 Range</span>
          </div>
        </div>

        <div className="stat-pod">
          <div className="stat-pod-icon teal">
            <BarChart3 size={24} />
          </div>
          <div className="stat-pod-info">
            <span className="stat-pod-label">Classes</span>
            <span className="stat-pod-value">10 (0–9)</span>
          </div>
        </div>
      </div>

      {/* Interactive Sample Inspector */}
      {activeSample && (
        <div className="card inspector-card">
          <div className="inspector-preview-col">
            <canvas
              width={128}
              height={128}
              className="gallery-canvas-large"
              ref={(el) => {
                if (el) {
                  const ctx = el.getContext('2d');
                  if (ctx) {
                    ctx.imageSmoothingEnabled = false;
                    const cellSize = 128 / 8;
                    ctx.fillStyle = '#060a12';
                    ctx.fillRect(0, 0, 128, 128);
                    for (let row = 0; row < 8; row++) {
                      for (let col = 0; col < 8; col++) {
                        const val = activeSample.image[row][col];
                        const intensity = Math.round((val / 16) * 255);
                        ctx.fillStyle = `rgb(${Math.round(intensity * 0.7)}, ${Math.round(intensity * 0.95)}, ${intensity})`;
                        ctx.fillRect(col * cellSize, row * cellSize, cellSize, cellSize);
                      }
                    }
                  }
                }
              }}
            />
            <div style={{ textAlign: 'center' }}>
              <span className="brand-sub" style={{ fontSize: '0.8rem' }}>
                Active Digit
              </span>
              <div style={{ fontSize: '1.75rem', fontWeight: 900, color: '#fff' }}>
                Digit {activeSample.label}
              </div>
            </div>
            {onTestSample && (
              <button
                type="button"
                className="btn btn-primary"
                style={{ fontSize: '0.82rem', padding: '0.5rem 1rem' }}
                onClick={() => onTestSample(activeSample)}
              >
                <Sparkles size={15} /> Test in Recognizer
              </button>
            )}
          </div>

          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
              <div>
                <h3 className="card-title">
                  <Layers size={18} color="var(--cyan-glow)" /> 64-Feature Pixel Matrix (0 to 16)
                </h3>
                <p className="card-desc" style={{ marginBottom: 0 }}>
                  Hover over any pixel cell to inspect its exact grayscale numerical intensity.
                </p>
              </div>
            </div>

            <div className="pixel-grid-inspector">
              {activeSample.flat.map((val, idx) => {
                const ratio = Math.max(0, Math.min(16, val)) / 16;
                const r = Math.round(ratio * 34);
                const g = Math.round(ratio * 211);
                const b = Math.round(ratio * 238);
                const isLight = ratio > 0.45;

                return (
                  <div
                    key={idx}
                    className="pixel-cell"
                    style={{
                      backgroundColor: `rgb(${r}, ${g}, ${b})`,
                      color: isLight ? '#060a12' : '#94a3b8',
                    }}
                    title={`Row ${Math.floor(idx / 8)}, Col ${idx % 8} · Intensity: ${val}/16`}
                  >
                    {val > 0 ? Math.round(val) : ''}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Class distribution chart */}
      <div className="card">
        <h2 className="card-title">
          <BarChart3 size={18} color="var(--cyan-glow)" /> Class Sample Distribution
        </h2>
        <p className="card-desc">Evenly stratified representation across all 10 digits in the dataset</p>
        <div style={{ position: 'relative', height: '240px' }}>
          <canvas ref={chartCanvasRef} />
        </div>
      </div>

      {/* Gallery with filter tabs */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem', marginBottom: '1.25rem' }}>
          <div>
            <h2 className="card-title">
              <Image size={18} color="var(--cyan-glow)" /> Sample Image Gallery
            </h2>
            <p className="card-desc" style={{ marginBottom: 0 }}>
              Click any image to load it into the 64-feature Matrix Inspector above
            </p>
          </div>

          <div className="filter-tabs-bar">
            <button
              type="button"
              className={`filter-tab ${selectedDigit === 'all' ? 'active' : ''}`}
              onClick={() => setSelectedDigit('all')}
            >
              All Digits
            </button>
            {Array.from({ length: 10 }, (_, d) => (
              <button
                key={d}
                type="button"
                className={`filter-tab ${selectedDigit === d ? 'active' : ''}`}
                onClick={() => setSelectedDigit(d)}
              >
                Digit {d}
              </button>
            ))}
          </div>
        </div>

        <div className="gallery-grid">
          {filteredSamples.map((s, idx) => {
            const isSelected = activeSample === s;
            return (
              <div
                key={idx}
                className={`gallery-card ${isSelected ? 'selected' : ''}`}
                onClick={() => setActiveSample(s)}
              >
                <canvas
                  width={64}
                  height={64}
                  className="gallery-canvas"
                  ref={(el) => {
                    if (el) {
                      const ctx = el.getContext('2d');
                      if (ctx) {
                        ctx.imageSmoothingEnabled = false;
                        const cellSize = 64 / 8;
                        ctx.fillStyle = '#060a12';
                        ctx.fillRect(0, 0, 64, 64);
                        for (let row = 0; row < 8; row++) {
                          for (let col = 0; col < 8; col++) {
                            const val = s.image[row][col];
                            const intensity = Math.round((val / 16) * 255);
                            ctx.fillStyle = `rgb(${Math.round(intensity * 0.7)}, ${Math.round(intensity * 0.95)}, ${intensity})`;
                            ctx.fillRect(col * cellSize, row * cellSize, cellSize, cellSize);
                          }
                        }
                      }
                    }
                  }}
                />
                <span className="gallery-badge">
                  {isSelected && <CheckCircle2 size={12} style={{ display: 'inline', marginRight: 3 }} />}
                  Digit {s.label}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
};
