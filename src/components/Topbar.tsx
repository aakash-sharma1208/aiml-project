import React from 'react';
import { Menu, Sparkles } from 'lucide-react';

interface TopbarProps {
  onToggleSidebar: () => void;
}

export const Topbar: React.FC<TopbarProps> = ({ onToggleSidebar }) => {
  return (
    <div className="topbar">
      <div className="topbar-brand">
        <div className="brand-icon" style={{ width: '32px', height: '32px' }}>
          <Sparkles size={16} />
        </div>
        <span className="topbar-title">DigiSVM</span>
      </div>

      <button
        type="button"
        className="btn btn-ghost"
        style={{ padding: '0.45rem', borderRadius: '8px' }}
        aria-label="Toggle menu"
        onClick={onToggleSidebar}
      >
        <Menu size={20} />
      </button>
    </div>
  );
};
