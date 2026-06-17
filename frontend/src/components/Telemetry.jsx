import React from 'react';

export default function Telemetry({ runs }) {
  const totalRuns = runs.length;
  const champion = runs.find(r => r.is_champion);
  const avgDuration = totalRuns > 0
    ? (runs.reduce((s, r) => s + r.duration_seconds, 0) / totalRuns).toFixed(1)
    : '0.0';
  const modelTypes = new Set(runs.map(r => r.model_type)).size;

  return (
    <div className="right-section" id="telemetry-panel">
      <div className="right-section__title">Telemetry</div>
      <div className="telemetry-grid">
        <div className="telemetry-cell">
          <div className="telemetry-cell__value">{totalRuns}</div>
          <div className="telemetry-cell__label">Total Runs</div>
        </div>
        <div className="telemetry-cell">
          <div className="telemetry-cell__value">{modelTypes}</div>
          <div className="telemetry-cell__label">Model Types</div>
        </div>
        <div className="telemetry-cell">
          <div className="telemetry-cell__value">{avgDuration}s</div>
          <div className="telemetry-cell__label">Avg Duration</div>
        </div>
        <div className="telemetry-cell">
          <div className="telemetry-cell__value">{champion ? '✓' : '—'}</div>
          <div className="telemetry-cell__label">Champion</div>
        </div>
      </div>
    </div>
  );
}
