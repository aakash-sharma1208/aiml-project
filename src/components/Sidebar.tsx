import React from 'react';
import { NavPage } from '../types';
import { PenTool, Database, Activity, Sparkles } from 'lucide-react';

interface SidebarProps {
  currentPage: NavPage;
  onSelectPage: (page: NavPage) => void;
  isOpen: boolean;
  onClose: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentPage,
  onSelectPage,
  isOpen,
  onClose,
}) => {
  const handleNavClick = (page: NavPage) => {
    onSelectPage(page);
    onClose();
  };

  return (
    <aside className={`sidebar ${isOpen ? 'open' : ''}`} id="sidebar">
      <div className="sidebar-header">
        <div className="brand-icon">
          <Sparkles size={22} />
        </div>
        <div className="brand-text">
          <span className="brand-name">DigiSVM</span>
          <span className="brand-sub">Neural Vector AI</span>
        </div>
      </div>

      <nav className="sidebar-nav">
        <button
          type="button"
          className={`nav-item ${currentPage === 'recognition' ? 'active' : ''}`}
          onClick={() => handleNavClick('recognition')}
        >
          <PenTool size={18} />
          Digit Recognition
          <span className="nav-badge">Live</span>
        </button>

        <button
          type="button"
          className={`nav-item ${currentPage === 'dataset' ? 'active' : ''}`}
          onClick={() => handleNavClick('dataset')}
        >
          <Database size={18} />
          Dataset Explorer
          <span className="nav-badge">1,797</span>
        </button>

        <button
          type="button"
          className={`nav-item ${currentPage === 'performance' ? 'active' : ''}`}
          onClick={() => handleNavClick('performance')}
        >
          <Activity size={18} />
          Model Performance
          <span className="nav-badge">99.7%</span>
        </button>
      </nav>

      <div className="sidebar-footer">
        <div className="status-dot-wrap">
          <div className="status-dot" />
          <span>RBF Kernel Engine</span>
        </div>
        <span style={{ color: 'var(--text-muted)' }}>v2.0</span>
      </div>
    </aside>
  );
};
