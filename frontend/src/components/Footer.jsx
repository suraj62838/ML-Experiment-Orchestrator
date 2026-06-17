import React from 'react';

export default function Footer({ runCount, championId }) {
  return (
    <footer className="footer app-footer" id="app-footer">
      <span>{runCount} experiments tracked</span>
      <span>
        champion: {championId ? championId.slice(0, 8) : 'none'}
      </span>
      <span>OrchestrML v1.0</span>
    </footer>
  );
}
