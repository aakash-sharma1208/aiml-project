import React, { useState, useEffect } from 'react';
import { NavPage, DatasetSample } from './types';
import { Sidebar } from './components/Sidebar';
import { Topbar } from './components/Topbar';
import { RecognitionPage } from './pages/RecognitionPage';
import { DatasetPage } from './pages/DatasetPage';
import { PerformancePage } from './pages/PerformancePage';

export const App: React.FC = () => {
  const [currentPage, setCurrentPage] = useState<NavPage>('recognition');
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(false);

  // Close sidebar on mobile when resizing to desktop
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth > 900) {
        setSidebarOpen(false);
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const handleTestSampleFromDataset = (_sample: DatasetSample) => {
    setCurrentPage('recognition');
  };

  return (
    <div className="app-container">
      {/* Cool glowing ambient orbs */}
      <div className="ambient-glow-1" />
      <div className="ambient-glow-2" />

      <Sidebar
        currentPage={currentPage}
        onSelectPage={(page) => setCurrentPage(page)}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <Topbar onToggleSidebar={() => setSidebarOpen((prev) => !prev)} />

      <main
        className="main-content"
        id="mainContent"
        onClick={() => {
          if (sidebarOpen && window.innerWidth <= 900) {
            setSidebarOpen(false);
          }
        }}
      >
        {currentPage === 'recognition' && <RecognitionPage />}
        {currentPage === 'dataset' && (
          <DatasetPage onTestSample={handleTestSampleFromDataset} />
        )}
        {currentPage === 'performance' && <PerformancePage />}
      </main>
    </div>
  );
};

export default App;
