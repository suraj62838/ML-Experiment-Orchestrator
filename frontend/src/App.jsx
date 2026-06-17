import React, { useState, useEffect, useCallback } from 'react';
import { listExperiments, getExperiment, promoteRun, deleteRun, healthCheck } from './api';

import Header from './components/Header';
import Footer from './components/Footer';
import RunList from './components/RunList';
import HeroPanel from './components/HeroPanel';
import MetricBars from './components/MetricBars';
import PipelineList from './components/PipelineList';
import HyperparamGrid from './components/HyperparamGrid';
import Telemetry from './components/Telemetry';
import ActivityFeed from './components/ActivityFeed';
import NewRunModal from './components/NewRunModal';
import PredictPanel from './components/PredictPanel';

export default function App() {
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [connected, setConnected] = useState(false);
  const [selectedId, setSelectedId] = useState(null);
  const [selectedRun, setSelectedRun] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');

  // ── Fetch runs ────────────────────────────────────────────────────────────
  const fetchRuns = useCallback(async () => {
    try {
      const data = await listExperiments();
      setRuns(data);
      setConnected(true);
    } catch {
      setConnected(false);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchRuns();
    const interval = setInterval(fetchRuns, 10000);
    return () => clearInterval(interval);
  }, [fetchRuns]);

  // ── Health check ──────────────────────────────────────────────────────────
  useEffect(() => {
    const check = async () => {
      try { await healthCheck(); setConnected(true); }
      catch { setConnected(false); }
    };
    check();
    const interval = setInterval(check, 15000);
    return () => clearInterval(interval);
  }, []);

  // ── Select run detail ─────────────────────────────────────────────────────
  useEffect(() => {
    if (!selectedId) { setSelectedRun(null); return; }
    getExperiment(selectedId)
      .then(setSelectedRun)
      .catch(() => setSelectedRun(null));
  }, [selectedId]);

  const champion = runs.find(r => r.is_champion) || null;

  const handlePromote = async (runId) => {
    try {
      await promoteRun(runId);
      await fetchRuns();
    } catch (err) {
      console.error('Promote failed:', err);
    }
  };

  const handleDelete = async (runId) => {
    try {
      await deleteRun(runId);
      if (selectedId === runId) setSelectedId(null);
      await fetchRuns();
    } catch (err) {
      console.error('Delete failed:', err);
    }
  };

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="app-layout" id="app-root">
      <Header connected={connected} />

      <RunList
        runs={runs}
        loading={loading}
        selectedId={selectedId}
        onSelect={setSelectedId}
        onNewRun={() => setShowModal(true)}
      />

      <div className="app-center" id="center-panel">
        <div className="center-tabs">
          {['overview', 'predict'].map(tab => (
            <button
              key={tab}
              className={`center-tab ${activeTab === tab ? 'center-tab--active' : ''}`}
              onClick={() => setActiveTab(tab)}
              id={`tab-${tab}`}
            >
              {tab}
            </button>
          ))}
        </div>

        {activeTab === 'overview' && (
          <div className="fade-in">
            <HeroPanel champion={champion} selected={selectedRun || champion} />

            <div className="section-header">Metrics</div>
            <MetricBars metrics={(selectedRun || champion)?.metrics} />

            <div className="section-header">Pipeline</div>
            <PipelineList steps={(selectedRun || champion)?.pipeline_summary} />

            <div className="section-header">Hyperparameters</div>
            <HyperparamGrid
              params={(selectedRun || champion)?.config?.model?.params}
            />

            {selectedRun && (
              <div style={{ marginTop: 24, display: 'flex', gap: 12 }}>
                {!selectedRun.is_champion && (
                  <button
                    className="predict-btn"
                    onClick={() => handlePromote(selectedRun.run_id)}
                    id="promote-btn"
                  >
                    Promote to Champion
                  </button>
                )}
                <button
                  onClick={() => handleDelete(selectedRun.run_id)}
                  style={{ borderColor: 'var(--accent-red)', color: 'var(--accent-red)' }}
                  id="delete-btn"
                >
                  Delete Run
                </button>
              </div>
            )}
          </div>
        )}

        {activeTab === 'predict' && <PredictPanel />}
      </div>

      <div className="app-right" id="right-panel">
        <Telemetry runs={runs} />
        <ActivityFeed runs={runs} />
      </div>

      <Footer runCount={runs.length} championId={champion?.run_id} />

      {showModal && (
        <NewRunModal
          onClose={() => setShowModal(false)}
          onSubmitted={fetchRuns}
        />
      )}
    </div>
  );
}
