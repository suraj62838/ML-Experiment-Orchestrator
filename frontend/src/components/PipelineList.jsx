import React from 'react';

export default function PipelineList({ steps }) {
  if (!steps || steps.length === 0) {
    return <div className="dim" style={{ fontSize: 'var(--fs-sm)' }}>No pipeline steps recorded.</div>;
  }

  return (
    <div className="pipeline-list" id="pipeline-list">
      {steps.map((step, i) => (
        <div className="pipeline-step" key={i}>
          <span className="pipeline-step__index">{String(i + 1).padStart(2, '0')}</span>
          <span className="pipeline-step__name">{step.type || step.step || step}</span>
          <span className="pipeline-step__tag tag">
            {step.cols ? (Array.isArray(step.cols) ? step.cols.join(', ') : step.cols) : ''}
          </span>
        </div>
      ))}
    </div>
  );
}
