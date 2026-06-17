/**
 * api.js — Centralized API client for the OrchestrML backend.
 */

const BASE = '/api';

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

/* ── Experiments ─────────────────────────────────────────────────────────── */

export async function listExperiments(limit = 50) {
  return request(`/experiments/?limit=${limit}`);
}

export async function getExperiment(runId) {
  return request(`/experiments/${runId}`);
}

export async function submitRun(config = {}, tune = false, nTrials = 20, metric = 'f1') {
  return request('/experiments/', {
    method: 'POST',
    body: JSON.stringify({ config, tune, n_trials: nTrials, metric }),
  });
}

export async function getRunStatus(runId) {
  return request(`/experiments/${runId}/status`);
}

export async function deleteRun(runId) {
  return request(`/experiments/${runId}`, { method: 'DELETE' });
}

export async function promoteRun(runId) {
  return request(`/experiments/${runId}/promote`, { method: 'POST' });
}

export async function compareRuns(runIdA, runIdB) {
  return request(`/experiments/compare/${runIdA}/${runIdB}`);
}

/* ── Datasets ────────────────────────────────────────────────────────────── */

export async function listDatasets() {
  return request('/datasets/');
}

export async function uploadDataset(file) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${BASE}/datasets/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function previewDataset(filename, rows = 20) {
  return request(`/datasets/preview/${filename}?rows=${rows}`);
}

export async function deleteDataset(filename) {
  return request(`/datasets/${filename}`, { method: 'DELETE' });
}

/* ── Pipeline ────────────────────────────────────────────────────────────── */

export async function validatePipeline(steps) {
  return request('/pipeline/validate', {
    method: 'POST',
    body: JSON.stringify({ steps }),
  });
}

/* ── Predict ─────────────────────────────────────────────────────────────── */

export async function predict(data) {
  return request('/predict/', {
    method: 'POST',
    body: JSON.stringify({ data }),
  });
}

/* ── Drift ───────────────────────────────────────────────────────────────── */

export async function getLatestDrift() {
  return request('/drift/latest');
}

/* ── Health ───────────────────────────────────────────────────────────────── */

export async function healthCheck() {
  return request('/health');
}
