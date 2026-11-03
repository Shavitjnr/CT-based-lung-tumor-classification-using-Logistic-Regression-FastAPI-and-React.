import React, { useState } from 'react';
import { 
  CheckCircle2, 
  AlertTriangle, 
  ArrowLeft, 
  Sliders, 
  Layers, 
  HelpCircle, 
  TrendingUp, 
  TrendingDown, 
  BarChart2, 
  Download,
  Search,
  ExternalLink
} from 'lucide-react';
import { PredictionResult, SampleScan } from '../services/api';

interface PredictionResultsProps {
  result: PredictionResult;
  previewUrl: string;
  sampleInfo?: SampleScan;
  onBackToUpload: () => void;
  onNavigateToPerformance: () => void;
}

export const PredictionResults: React.FC<PredictionResultsProps> = ({
  result,
  previewUrl,
  sampleInfo,
  onBackToUpload,
  onNavigateToPerformance,
}) => {
  const [featureSearch, setFeatureSearch] = useState('');
  const [activeTab, setActiveTab] = useState<'contributions' | 'all_features'>('contributions');

  const isMalignant = result.predicted_class === 'Malignant';
  const malignantProb = result.probabilities.malignant;
  const benignProb = result.probabilities.benign;

  const filteredFeatures = Object.entries(result.extracted_features).filter(([key]) =>
    key.toLowerCase().includes(featureSearch.toLowerCase())
  );

  return (
    <div className="space-y-8 pb-12">
      {/* Action Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-300 pb-4">
        <div>
          <button
            onClick={onBackToUpload}
            className="text-xs font-mono text-teal-700 hover:text-teal-900 flex items-center space-x-1.5 mb-1 cursor-pointer"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Analyze Another CT Scan</span>
          </button>
          <h1 className="text-2xl font-extrabold text-slate-900 font-mono tracking-tight">
            Diagnostic Classification Report
          </h1>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={onNavigateToPerformance}
            className="px-3.5 py-2 bg-slate-800 hover:bg-slate-700 text-white text-xs font-mono font-semibold flex items-center space-x-1.5 border border-slate-700 cursor-pointer"
          >
            <BarChart2 className="w-4 h-4 text-teal-400" />
            <span>View ROC/PR Curves</span>
          </button>
        </div>
      </div>

      {/* Main Classification Banner */}
      <div className={`p-6 border-2 ${
        isMalignant
          ? 'bg-rose-50 border-rose-500 text-rose-950'
          : 'bg-emerald-50 border-emerald-500 text-emerald-950'
      }`}>
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <span className={`px-2.5 py-1 text-xs font-bold font-mono uppercase tracking-wider text-white ${
                isMalignant ? 'bg-rose-700' : 'bg-emerald-700'
              }`}>
                Classification: {result.predicted_class}
              </span>
              <span className="text-xs font-mono text-slate-600">
                Threshold Applied: {result.threshold_applied.toFixed(2)}
              </span>
            </div>

            <div className="text-3xl sm:text-4xl font-extrabold font-mono tracking-tight">
              {isMalignant ? 'Malignant Pulmonary Carcinoma' : 'Benign Pulmonary Nodule'}
            </div>

            <p className="text-xs text-slate-700 max-w-2xl leading-relaxed">
              {result.explanation}
            </p>
          </div>

          {/* Probability Gauge / Metrics */}
          <div className="bg-white border border-slate-300 p-4 min-w-[280px] space-y-3">
            <div className="text-xs font-bold uppercase tracking-wider text-slate-600 font-mono">
              Estimated Class Probabilities
            </div>

            {/* Progress Bars */}
            <div className="space-y-2 font-mono">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-rose-700 font-semibold">Malignant:</span>
                  <span className="font-bold text-rose-900">{(malignantProb * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full h-3 bg-slate-100 border border-slate-200 overflow-hidden">
                  <div
                    className="h-full bg-rose-600 transition-all duration-500"
                    style={{ width: `${malignantProb * 100}%` }}
                  ></div>
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-emerald-700 font-semibold">Benign:</span>
                  <span className="font-bold text-emerald-900">{(benignProb * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full h-3 bg-slate-100 border border-slate-200 overflow-hidden">
                  <div
                    className="h-full bg-emerald-600 transition-all duration-500"
                    style={{ width: `${benignProb * 100}%` }}
                  ></div>
                </div>
              </div>
            </div>

            <div className="pt-2 border-t border-slate-200 flex justify-between text-[11px] font-mono text-slate-500">
              <span>Raw Model Logit:</span>
              <span className="font-bold text-slate-800">{result.raw_logit.toFixed(3)}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Two Column Layout: Image & Features */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: CT Image & Preprocessing Metadata */}
        <div className="lg:col-span-4 space-y-6">
          <div className="bg-white border border-slate-300 p-5 space-y-4">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800 font-mono border-b border-slate-200 pb-2">
              Thoracic CT Slice Preview
            </h3>

            <div className="relative aspect-square w-full bg-black border border-slate-800 overflow-hidden flex items-center justify-center">
              <img
                src={previewUrl}
                alt="Analyzed CT Scan"
                className="max-h-full max-w-full object-contain"
              />
              <div className="absolute top-2 left-2 px-2 py-0.5 bg-black/80 text-[10px] text-teal-300 font-mono border border-slate-700">
                Axial View (256x256 CLAHE)
              </div>
            </div>

            {/* Scan Metadata */}
            <div className="text-xs font-mono space-y-1.5 bg-slate-50 p-3 border border-slate-200">
              <div className="flex justify-between">
                <span className="text-slate-500">Original Dimensions:</span>
                <span className="text-slate-800">{result.preprocessed_stats.raw_shape.join(' × ')}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Resampled Dimension:</span>
                <span className="text-slate-800">{result.preprocessed_stats.target_size.join(' × ')}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Parenchyma Cavity Ratio:</span>
                <span className="text-slate-800">{(result.preprocessed_stats.parenchyma_ratio * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Model Pipeline:</span>
                <span className="text-teal-700 font-bold">StandardScaler + LR v{result.model_version}</span>
              </div>
              {sampleInfo && (
                <div className="flex justify-between pt-1 border-t border-slate-200">
                  <span className="text-slate-500">Ground Truth Match:</span>
                  <span className={`font-bold ${sampleInfo.true_class === result.predicted_class ? 'text-emerald-700' : 'text-rose-700'}`}>
                    {sampleInfo.true_class === result.predicted_class ? '✓ Verified Concordant' : '✗ Discordant'}
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Column: Radiomic Explainability & Feature Table */}
        <div className="lg:col-span-8 space-y-6">
          <div className="bg-white border border-slate-300 p-5 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 pb-3">
              <div className="flex items-center space-x-2">
                <Layers className="w-4 h-4 text-teal-600" />
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800 font-mono">
                  Radiomic Feature Explainability
                </h3>
              </div>

              {/* Tabs */}
              <div className="flex space-x-1 font-mono text-xs">
                <button
                  onClick={() => setActiveTab('contributions')}
                  className={`px-3 py-1.5 border transition-all cursor-pointer ${
                    activeTab === 'contributions'
                      ? 'bg-teal-700 text-white border-teal-700 font-semibold'
                      : 'bg-slate-100 text-slate-600 border-slate-300 hover:bg-slate-200'
                  }`}
                >
                  Top Driving Features
                </button>
                <button
                  onClick={() => setActiveTab('all_features')}
                  className={`px-3 py-1.5 border transition-all cursor-pointer ${
                    activeTab === 'all_features'
                      ? 'bg-teal-700 text-white border-teal-700 font-semibold'
                      : 'bg-slate-100 text-slate-600 border-slate-300 hover:bg-slate-200'
                  }`}
                >
                  All 21 Features Table
                </button>
              </div>
            </div>

            {/* Tab 1: Top Driving Features (Waterfall / Bar list) */}
            {activeTab === 'contributions' && (
              <div className="space-y-4">
                <p className="text-xs text-slate-600">
                  Decomposition of the Logistic Regression logit: each feature contributes a log-odds shift equal to 
                  <code className="bg-slate-100 px-1 py-0.5 text-slate-800 ml-1">Z-score × Coefficient</code>.
                </p>

                <div className="space-y-2.5">
                  {result.top_contributions.map((feat, i) => {
                    const pushesMalignant = feat.logit_impact > 0;
                    return (
                      <div
                        key={feat.feature}
                        className="p-3 border border-slate-200 bg-slate-50 flex flex-col sm:flex-row sm:items-center justify-between gap-2"
                      >
                        <div className="space-y-0.5">
                          <div className="flex items-center space-x-2 font-mono">
                            <span className="text-xs font-bold text-slate-900">{feat.feature}</span>
                            <span className={`text-[10px] font-semibold px-1.5 py-0.5 border ${
                              pushesMalignant
                                ? 'bg-rose-100 text-rose-800 border-rose-300'
                                : 'bg-emerald-100 text-emerald-800 border-emerald-300'
                            }`}>
                              {feat.direction}
                            </span>
                          </div>
                          <p className="text-[11px] text-slate-500">{feat.description}</p>
                        </div>

                        <div className="flex items-center space-x-4 text-xs font-mono shrink-0">
                          <div>
                            <span className="text-slate-400 text-[10px] block">Raw Val:</span>
                            <span className="text-slate-800 font-semibold">{feat.raw_value}</span>
                          </div>
                          <div>
                            <span className="text-slate-400 text-[10px] block">Z-Score:</span>
                            <span className="text-slate-800 font-semibold">{feat.z_score}</span>
                          </div>
                          <div>
                            <span className="text-slate-400 text-[10px] block">Impact:</span>
                            <span className={`font-bold ${pushesMalignant ? 'text-rose-700' : 'text-emerald-700'}`}>
                              {feat.logit_impact > 0 ? `+${feat.logit_impact}` : feat.logit_impact}
                            </span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Tab 2: All 21 Features Table */}
            {activeTab === 'all_features' && (
              <div className="space-y-3">
                <div className="relative">
                  <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-2.5" />
                  <input
                    type="text"
                    placeholder="Search feature by name (e.g., glcm, lbp, intensity, edge)..."
                    value={featureSearch}
                    onChange={(e) => setFeatureSearch(e.target.value)}
                    className="w-full pl-9 pr-3 py-1.5 text-xs font-mono border border-slate-300 focus:border-teal-600 focus:outline-none"
                  />
                </div>

                <div className="overflow-x-auto border border-slate-200">
                  <table className="w-full text-left text-xs font-mono">
                    <thead className="bg-slate-100 border-b border-slate-200 text-slate-600 uppercase text-[10px]">
                      <tr>
                        <th className="p-2.5">Feature Name</th>
                        <th className="p-2.5">Extracted Value</th>
                        <th className="p-2.5">Biological / Physical Rationale</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200">
                      {filteredFeatures.map(([key, val]) => {
                        const contrib = result.feature_contributions.find(c => c.feature === key);
                        return (
                          <tr key={key} className="hover:bg-slate-50">
                            <td className="p-2.5 font-bold text-slate-900">{key}</td>
                            <td className="p-2.5 text-teal-800 font-semibold">{val}</td>
                            <td className="p-2.5 text-slate-600 text-[11px] font-sans">
                              {contrib?.description || 'Quantitative radiomic descriptor'}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Mandatory Medical Warning */}
      <div className="bg-rose-50 border-2 border-rose-300 p-5 flex items-start space-x-3.5">
        <AlertTriangle className="w-5 h-5 text-rose-700 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <h4 className="text-xs font-bold uppercase tracking-wider text-rose-900 font-mono">
            Clinical Warning & Decision Limitations
          </h4>
          <p className="text-xs text-rose-800 leading-relaxed font-sans">
            "This result is produced by an experimental machine-learning model. It is not a medical diagnosis 
            and must not be used to make treatment decisions." Non-contrast CT slices may exhibit overlapping radiomic 
            profiles with infectious granulomas, hamartomas, or organizing pneumonia. Histopathological biopsy remains the gold standard.
          </p>
        </div>
      </div>
    </div>
  );
};

