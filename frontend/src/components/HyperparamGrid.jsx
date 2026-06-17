import React from 'react';

export default function HyperparamGrid({ params }) {
  if (!params || Object.keys(params).length === 0) {
    return <div className="dim" style={{ fontSize: 'var(--fs-sm)' }}>No hyperparameters recorded.</div>;
  }

  const entries = Object.entries(params);

  return (
    <div className="hyperparam-grid" id="hyperparam-grid">
      {entries.map(([key, val]) => (
        <div className="hyperparam-cell" key={key}>
          <div className="hyperparam-cell__key">{key}</div>
          <div className="hyperparam-cell__val">
            {typeof val === 'number' ? val.toPrecision(4) : String(val)}
          </div>
        </div>
      ))}
    </div>
  );
}
