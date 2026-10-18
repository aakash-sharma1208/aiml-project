import React, { useEffect, useRef } from 'react';
import { Chart, registerables } from 'chart.js';
import datasetData from '../data/dataset.json';
import { DatasetInfo } from '../types';
import { Activity, CheckCircle, Target, Zap, BarChart2 } from 'lucide-react';

Chart.register(...registerables);

const data = datasetData as unknown as DatasetInfo;

export const PerformancePage: React.FC = () => {
  const chartCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const chartInstanceRef = useRef<Chart | null>(null);

  const meta = data.metadata;
  const perClass = meta.per_class;
  const cm = meta.confusion_matrix;
  const pct = (v?: number) => (v !== undefined ? `${(v * 100).toFixed(2)}%` : '—');

  useEffect(() => {
    const canvas = chartCanvasRef.current;
    if (!canvas) return;

    if (chartInstanceRef.current) {
      chartInstanceRef.current.destroy();
      chartInstanceRef.current = null;
    }

    const labels = Array.from({ length: 10 }, (_, i) => `Digit ${i}`);

    chartInstanceRef.current = new Chart(canvas, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'Precision',
            data: perClass.precision.map((v) => Number((v * 100).toFixed(2))),
            backgroundColor: 'rgba(6, 182, 212, 0.75)',
            borderColor: 'rgba(34, 211, 238, 1)',
            borderWidth: 1,
            borderRadius: 4,
          },
          {
            label: 'Recall',
            data: perClass.recall.map((v) => Number((v * 100).toFixed(2))),
            backgroundColor: 'rgba(59, 130, 246, 0.75)',
            borderColor: 'rgba(96, 165, 250, 1)',
            borderWidth: 1,
            borderRadius: 4,
          },
          {
            label: 'F1-Score',
            data: perClass.f1.map((v) => Number((v * 100).toFixed(2))),
            backgroundColor: 'rgba(20, 184, 166, 0.75)',
            borderColor: 'rgba(45, 212, 191, 1)',
            borderWidth: 1,
            borderRadius: 4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
          duration: 750,
          easing: 'easeOutQuart',
        },
        plugins: {
          legend: {
            labels: {
              color: '#94a3b8',
              font: { size: 12, weight: 'bold' },
            },
          },
          tooltip: {
            backgroundColor: 'rgba(12, 19, 34, 0.9)',
            titleColor: '#38bdf8',
            bodyColor: '#f8fafc',
            borderColor: 'rgba(56, 189, 248, 0.3)',
            borderWidth: 1,
            padding: 10,
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: ${(ctx.parsed.y ?? 0).toFixed(2)}%`,
            },
          },
        },
        scales: {
          x: {
            ticks: { color: '#94a3b8', font: { weight: 'bold' } },
            grid: { color: 'rgba(255,255,255,0.04)' },
          },
          y: {
            min: 80,
            max: 100,
            ticks: {
              color: '#94a3b8',
              callback: (v) => `${v}%`,
            },
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
  }, [perClass]);

  const n = cm.length;

  return (
    <section className="page" id="page-performance">
      <div className="page-header">
        <span className="header-tag">
          <Activity size={14} /> Model Validation &amp; Metrics
        </span>
        <h1>Model Evaluation &amp; Accuracy</h1>
        <p>
          Empirical validation metrics computed on the 360 held-out test samples (20% split) never
          seen during model training.
        </p>
      </div>

      {/* Metrics Row */}
      <div className="metrics-row">
        <div className="metric-highlight-card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span className="stat-pod-label">Test Accuracy</span>
            <CheckCircle size={18} color="var(--cyan-glow)" />
          </div>
          <div className="metric-highlight-val">{pct(meta.accuracy)}</div>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
            359 of 360 test samples correct
          </span>
        </div>

        <div className="metric-highlight-card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span className="stat-pod-label">Precision (Macro)</span>
            <Target size={18} color="var(--sky-bright)" />
          </div>
          <div className="metric-highlight-val">{pct(meta.precision)}</div>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
            Average precision across classes 0–9
          </span>
        </div>

        <div className="metric-highlight-card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span className="stat-pod-label">Recall (Macro)</span>
            <Zap size={18} color="var(--indigo-cool)" />
          </div>
          <div className="metric-highlight-val">{pct(meta.recall)}</div>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
            Average recall across classes 0–9
          </span>
        </div>

        <div className="metric-highlight-card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span className="stat-pod-label">F1-Score (Macro)</span>
            <Activity size={18} color="var(--teal-cool)" />
          </div>
          <div className="metric-highlight-val">{pct(meta.f1_score)}</div>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
            Harmonic balance of precision &amp; recall
          </span>
        </div>
      </div>

      {/* Per-class Chart */}
      <div className="card">
        <h2 className="card-title">
          <BarChart2 size={18} color="var(--cyan-glow)" /> Per-Class Precision, Recall &amp; F1
        </h2>
        <p className="card-desc">Individual benchmark performance breakdown for each digit class (0 through 9)</p>
        <div style={{ position: 'relative', height: '280px' }}>
          <canvas ref={chartCanvasRef} />
        </div>
      </div>

      {/* Confusion Matrix Table */}
      <div className="card">
        <h2 className="card-title">
          <Activity size={18} color="var(--cyan-glow)" /> 10×10 Confusion Matrix
        </h2>
        <p className="card-desc">
          Actual classes vs. Predicted classes on test set. Diagonal indicates true positives.
        </p>
        <div style={{ overflowX: 'auto', marginTop: '0.75rem' }}>
          <table className="cm-table-cool">
            <thead>
              <tr>
                <th>Actual \ Predicted</th>
                {Array.from({ length: n }, (_, j) => (
                  <th key={j}>Digit {j}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {cm.map((row, i) => (
                <tr key={i}>
                  <td className="cm-row-header">Digit {i}</td>
                  {row.map((val, j) => {
                    const isDiag = i === j;
                    const isZero = val === 0;
                    const cellClass = isDiag
                      ? 'cm-diag-cool'
                      : isZero
                      ? 'cm-zero-cool'
                      : 'cm-err-cool';

                    return (
                      <td
                        key={j}
                        className={cellClass}
                        title={`Actual Digit ${i}, Predicted Digit ${j}: ${val} instances`}
                      >
                        {val}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
};
