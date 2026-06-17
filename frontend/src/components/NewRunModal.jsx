import React, { useState } from 'react';
import { submitRun, uploadDataset } from '../api';

export default function NewRunModal({ onClose, onSubmitted }) {
  const [modelType, setModelType] = useState('logistic_regression');
  const [tune, setTune] = useState(false);
  const [nTrials, setNTrials] = useState(20);
  const [metric, setMetric] = useState('f1');
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState('idle');
  const [error, setError] = useState('');

  const handleSubmit = async () => {
    setStatus('submitting');
    setError('');
    try {
      if (file) {
        await uploadDataset(file);
      }
      const config = {
        model: { type: modelType, params: {} },
      };
      if (file) {
        config.data = { filepath: `uploads/${file.name}`, target_col: 'target' };
      }
      await submitRun(config, tune, nTrials, metric);
      setStatus('submitted');
      onSubmitted?.();
      setTimeout(onClose, 800);
    } catch (err) {
      setError(err.message);
      setStatus('error');
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose} id="new-run-modal">
      <div className="modal" onClick={e => e.stopPropagation()}>
        <div className="modal__header">
          <span className="modal__title">New Experiment</span>
          <button className="modal__close" onClick={onClose}>×</button>
        </div>
        <div className="modal__body">
          <div className="form-field">
            <label className="form-field__label">Dataset (CSV)</label>
            <input
              type="file"
              accept=".csv"
              onChange={e => setFile(e.target.files[0])}
              id="dataset-upload-input"
            />
          </div>
          <div className="form-field">
            <label className="form-field__label">Model Type</label>
            <select value={modelType} onChange={e => setModelType(e.target.value)} id="model-type-select">
              <option value="logistic_regression">Logistic Regression</option>
              <option value="random_forest">Random Forest</option>
              <option value="gradient_boosting">Gradient Boosting</option>
              <option value="svm">SVM</option>
              <option value="knn">KNN</option>
            </select>
          </div>
          <div className="form-field">
            <label className="form-field__label" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <input
                type="checkbox"
                checked={tune}
                onChange={e => setTune(e.target.checked)}
                style={{ width: 'auto' }}
                id="tune-checkbox"
              />
              Enable Hyperparameter Tuning
            </label>
          </div>
          {tune && (
            <>
              <div className="form-field">
                <label className="form-field__label">Trials</label>
                <input
                  type="number"
                  value={nTrials}
                  onChange={e => setNTrials(parseInt(e.target.value) || 20)}
                  min={5}
                  max={200}
                  id="trials-input"
                />
              </div>
              <div className="form-field">
                <label className="form-field__label">Metric</label>
                <select value={metric} onChange={e => setMetric(e.target.value)} id="metric-select">
                  <option value="f1">F1</option>
                  <option value="accuracy">Accuracy</option>
                  <option value="precision">Precision</option>
                  <option value="recall">Recall</option>
                </select>
              </div>
            </>
          )}

          {status === 'submitting' && (
            <div className="progress-bar">
              <div className="progress-bar__fill progress-bar__fill--indeterminate" />
            </div>
          )}
          {status === 'submitted' && (
            <div className="tag tag--green" style={{ alignSelf: 'flex-start' }}>Submitted ✓</div>
          )}
          {error && <div className="tag tag--red" style={{ alignSelf: 'flex-start' }}>{error}</div>}
        </div>
        <div className="modal__footer">
          <button onClick={onClose}>Cancel</button>
          <button
            className="predict-btn"
            onClick={handleSubmit}
            disabled={status === 'submitting'}
            id="submit-run-btn"
          >
            {status === 'submitting' ? 'Running...' : 'Launch'}
          </button>
        </div>
      </div>
    </div>
  );
}
