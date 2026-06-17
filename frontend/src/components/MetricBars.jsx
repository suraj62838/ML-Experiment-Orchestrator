import React from 'react';

function getBarClass(value) {
  if (value < 0.5) return 'metric-row__bar-fill--red';
  if (value < 0.75) return 'metric-row__bar-fill--amber';
  return '';
}

export default function MetricBars({ metrics }) {
  if (!metrics) return null;

  const displayKeys = Object.keys(metrics).filter(
    k => k !== 'task' && k !== 'confusion_matrix'
  );

  return (
    <div className="metrics-grid" id="metric-bars">
      {displayKeys.map(key => {
        const val = metrics[key];
        const normalized = typeof val === 'number' ? Math.min(Math.abs(val), 1) : 0;
        return (
          <div className="metric-row" key={key}>
            <span className="metric-row__label">{key}</span>
            <div className="metric-row__bar-bg">
              <div
                className={`metric-row__bar-fill ${getBarClass(normalized)}`}
                style={{ width: `${normalized * 100}%` }}
              />
            </div>
            <span className="metric-row__value">
              {typeof val === 'number' ? val.toFixed(4) : val}
            </span>
          </div>
        );
      })}
    </div>
  );
}
