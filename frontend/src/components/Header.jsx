import React from 'react';

export default function Header({ connected }) {
  return (
    <header className="header app-header" id="app-header">
      <div className="header__brand">
        <span>Orchestr</span>ML
      </div>
      <div className="header__nav">
        <div className="header__status">
          <div className={`header__dot ${connected ? 'blink' : 'header__dot--disconnected'}`} />
          {connected ? 'connected' : 'disconnected'}
        </div>
      </div>
    </header>
  );
}
