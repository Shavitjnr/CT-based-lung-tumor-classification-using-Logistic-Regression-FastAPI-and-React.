import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { HomeDashboard } from './pages/HomeDashboard';
import { CTScanUpload } from './pages/CTScanUpload';
import { PredictionResults } from './pages/PredictionResults';
import { ModelPerformance } from './pages/ModelPerformance';
import { AboutProject } from './pages/AboutProject';
import { 
  fetchModelMetrics, 
  fetchModelInfo, 
  fetchSampleScans, 
  ModelMetricsResponse, 
  ModelInfoResponse, 
  SampleScan, 
  PredictionResult 
} from './services/api';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('home');
  const [metrics, setMetrics] = useState<ModelMetricsResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [samples, setSamples] = useState<SampleScan[]>([]);
  const [currentResult, setCurrentResult] = useState<PredictionResult | null>(null);
  const [currentPreviewUrl, setCurrentPreviewUrl] = useState<string | null>(null);
  const [currentSampleInfo, setCurrentSampleInfo] = useState<SampleScan | undefined>(undefined);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    async function loadInitialData() {
      try {
        const [metricsData, infoData, samplesData] = await Promise.all([
          fetchModelMetrics().catch(() => null),
          fetchModelInfo().catch(() => null),
          fetchSampleScans().catch(() => []),
        ]);
        if (metricsData) setMetrics(metricsData);
        if (infoData) setModelInfo(infoData);
        if (samplesData) setSamples(samplesData);
      } catch (err) {
        console.error('Failed to load initial telemetry:', err);
      } finally {
        setIsLoading(false);
      }
    }
    loadInitialData();
  }, []);

  const handlePredictionComplete = (
    result: PredictionResult, 
    previewUrl: string, 
    sampleInfo?: SampleScan
  ) => {
    setCurrentResult(result);
    setCurrentPreviewUrl(previewUrl);
    setCurrentSampleInfo(sampleInfo);
    setActiveTab('results');
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900 font-sans">
      {/* Navigation Bar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        hasResult={currentResult !== null}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 pt-8">
        {activeTab === 'home' && (
          <HomeDashboard
            metrics={metrics}
            modelInfo={modelInfo}
            onNavigate={(tab) => setActiveTab(tab)}
          />
        )}

        {activeTab === 'upload' && (
          <CTScanUpload
            samples={samples}
            onPredictionComplete={handlePredictionComplete}
          />
        )}

        {activeTab === 'results' && currentResult && currentPreviewUrl && (
          <PredictionResults
            result={currentResult}
            previewUrl={currentPreviewUrl}
            sampleInfo={currentSampleInfo}
            onBackToUpload={() => setActiveTab('upload')}
            onNavigateToPerformance={() => setActiveTab('performance')}
          />
        )}

        {activeTab === 'performance' && (
          <ModelPerformance
            metrics={metrics}
            modelInfo={modelInfo}
          />
        )}

        {activeTab === 'about' && <AboutProject />}
      </main>

      {/* Sharp Clinical Footer */}
      <footer className="bg-slate-900 border-t border-slate-800 text-slate-400 py-6 text-xs font-mono mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center space-x-2">
            <span className="w-2 h-2 bg-teal-500"></span>
            <span className="text-white font-semibold">LungScan AI</span>
            <span className="text-slate-500">|</span>
            <span>B.Tech AI & ML Capstone Project</span>
          </div>

          <div className="text-slate-400 text-center sm:text-right text-[11px]">
            Trained on IQ-OTH/NCCD Lung CT Dataset • Scikit-Learn Logistic Regression
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;

