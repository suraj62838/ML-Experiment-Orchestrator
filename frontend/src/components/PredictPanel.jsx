import React, { useState } from 'react';
import { predict } from '../api';

export default function PredictPanel() {
  const [input, setInput] = useState('[\n  {"gender": "male", "category": "A", "feature_1": 0.5, "feature_2": -1.2, "feature_3": 3.0, "feature_4": 2}\n]');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    setError('');
    setResult(null);
    setLoading(true);
    try {
      const data = JSON.parse(input);
      const res = await predict(data);
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="predict-panel fade-in" id="predict-panel">
      <div className="section-header">Predict — Deploy Workspace</div>

      <div className="predict-panel__input">
        <textarea
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder='Paste JSON array of records...'
          id="predict-input"
        />
      </div>

      <div className="predict-panel__actions">
        <button className="predict-btn" onClick={handlePredict} disabled={loading} id="predict-btn">
          {loading ? 'Predicting...' : 'Run Prediction'}
        </button>
      </div>

      {error && <div className="tag tag--red" style={{ marginBottom: 16 }}>{error}</div>}

      {result && (
        <>
          <div style={{ marginBottom: 8, fontSize: 'var(--fs-xs)', color: 'var(--text-tertiary)' }}>
            Model: {result.model_type} ({result.model_run_id.slice(0, 8)})
          </div>
          <table className="result-table" id="predict-results-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Prediction</th>
              </tr>
            </thead>
            <tbody>
              {result.predictions.map((p, i) => (
                <tr key={i}>
                  <td>{i + 1}</td>
                  <td>{String(p)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
