import React from 'react';

function relativeTime(isoStr) {
  const d = new Date(isoStr);
  const diff = (Date.now() - d.getTime()) / 1000;
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

export default function ActivityFeed({ runs }) {
  const recent = runs.slice(0, 15);

  return (
    <div className="right-section" id="activity-feed">
      <div className="right-section__title">Activity</div>
      <div className="activity-feed">
        {recent.map(run => (
          <div className="activity-item" key={run.run_id}>
            <div className={`activity-item__dot ${run.is_champion ? 'activity-item__dot--green' : ''}`} />
            <span className="activity-item__text">
              {run.run_id.slice(0, 8)} — {run.model_type}
            </span>
            <span className="activity-item__time">{relativeTime(run.timestamp)}</span>
          </div>
        ))}
        {recent.length === 0 && (
          <div className="dim" style={{ fontSize: 'var(--fs-xs)', padding: 'var(--sp-2) 0' }}>
            No activity yet
          </div>
        )}
      </div>
    </div>
  );
}
