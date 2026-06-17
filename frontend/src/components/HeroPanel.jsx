import React from 'react';

export default function HeroPanel({ champion, selected }) {
  const renderCard = (run, label, isChampion) => {
    if (!run) {
      return (
        <div className="hero-card">
          <div className="hero-card__label">{label}</div>
          <div className="dim" style={{ fontSize: 'var(--fs-sm)' }}>No run selected</div>
        </div>
      );
    }
    return (
      <div className={`hero-card ${isChampion ? 'hero-card--champion' : ''}`}>
        <div className="hero-card__label">{label}</div>
        <div className="hero-card__run-id">{run.run_id.slice(0, 8)}</div>
        <div className="hero-card__model">{run.model_type}</div>
        <div className="hero-card__duration">{run.duration_seconds.toFixed(1)}s runtime</div>
      </div>
    );
  };

  return (
    <div className="hero-grid" id="hero-panel">
      {renderCard(champion, 'Champion', true)}
      {renderCard(selected, 'Selected Run', false)}
    </div>
  );
}
