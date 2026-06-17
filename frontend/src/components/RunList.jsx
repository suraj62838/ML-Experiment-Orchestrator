import React from 'react';

function getMetricValue(metrics) {
  if (!metrics) return null;
  if (metrics.f1_weighted != null) return metrics.f1_weighted;
  if (metrics.accuracy != null) return metrics.accuracy;
  if (metrics.rmse != null) return metrics.rmse;
  const keys = Object.keys(metrics).filter(k => k !== 'task' && k !== 'confusion_matrix');
  return keys.length > 0 ? metrics[keys[0]] : null;
}

function SkeletonRow() {
  return (
    <div className="skeleton-row">
      <div className="skeleton-row__circle skeleton" />
      <div className="skeleton-row__lines">
        <div className="skeleton" style={{ width: '60%' }} />
        <div className="skeleton" style={{ width: '40%' }} />
      </div>
    </div>
  );
}

export default function RunList({ runs, loading, selectedId, onSelect, onNewRun }) {
  const champion = runs.find(r => r.is_champion);
  const others = runs.filter(r => !r.is_champion);

  return (
    <div className="app-left" id="run-list-panel">
      <button className="new-run-btn" onClick={onNewRun} id="new-run-btn">
        + New Experiment
      </button>

      <div className="left-panel-header">
        <span className="left-panel-header__title">Experiments</span>
        <span className="left-panel-header__count">{runs.length}</span>
      </div>

      {loading && [0,1,2,3,4].map(i => <SkeletonRow key={i} />)}

      {!loading && champion && (
        <div
          className={`run-row run-row--champion ${selectedId === champion.run_id ? 'run-row--selected' : ''}`}
          onClick={() => onSelect(champion.run_id)}
          id={`run-row-${champion.run_id}`}
        >
          <div className="run-row__indicator run-row__indicator--ok" />
          <div className="run-row__info">
            <div className="run-row__id">
              {champion.run_id.slice(0, 8)}
              <span className="champion-badge" style={{ marginLeft: 8 }}>champion</span>
            </div>
            <div className="run-row__meta">
              {champion.model_type} · {champion.duration_seconds.toFixed(1)}s
            </div>
          </div>
          <div className="run-row__metric">
            {getMetricValue(champion.metrics)?.toFixed(4) ?? '—'}
          </div>
        </div>
      )}

      {!loading && others.map(run => {
        const mv = getMetricValue(run.metrics);
        return (
          <div
            key={run.run_id}
            className={`run-row ${selectedId === run.run_id ? 'run-row--selected' : ''}`}
            onClick={() => onSelect(run.run_id)}
            id={`run-row-${run.run_id}`}
          >
            <div className="run-row__indicator run-row__indicator--ok" />
            <div className="run-row__info">
              <div className="run-row__id">{run.run_id.slice(0, 8)}</div>
              <div className="run-row__meta">
                {run.model_type} · {run.duration_seconds.toFixed(1)}s
              </div>
            </div>
            <div className="run-row__metric">{mv?.toFixed(4) ?? '—'}</div>
          </div>
        );
      })}

      {!loading && runs.length === 0 && (
        <div className="empty-state" style={{ padding: 40 }}>
          <div className="empty-state__icon">∅</div>
          <div>No experiments yet</div>
        </div>
      )}
    </div>
  );
}
