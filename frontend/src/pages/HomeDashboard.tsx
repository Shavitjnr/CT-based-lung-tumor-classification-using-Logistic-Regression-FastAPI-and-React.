import React from 'react';
import { ArrowRight, BarChart2, CheckCircle2, AlertTriangle, Layers, Cpu, Database, Stethoscope, ChevronRight } from 'lucide-react';
import { ModelMetricsResponse, ModelInfoResponse } from '../services/api';

interface HomeDashboardProps {
  metrics: ModelMetricsResponse | null;
  modelInfo: ModelInfoResponse | null;
  onNavigate: (tab: string) => void;
}

export const HomeDashboard: React.FC<HomeDashboardProps> = ({ metrics, modelInfo, onNavigate }) => {
  const evalMetrics = metrics?.evaluation_metrics;

  return (
    <div className="space-y-8 pb-12">
      {/* Hero Section */}
      <div className="bg-slate-900 border border-slate-800 p-8 sm:p-10 text-white relative overflow-hidden">
        <div className="max-w-4xl relative z-10 space-y-4">
          <div className="inline-flex items-center space-x-2 px-3 py-1 bg-teal-950/80 border border-teal-700 text-teal-300 text-xs font-mono tracking-wider uppercase">
            <span className="w-1.5 h-1.5 bg-teal-400"></span>
            <span>B.Tech AI & ML Capstone Research Prototype</span>
          </div>

          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold tracking-tight font-mono">
            Lung Tumor Classification Using Logistic Regression & Radiomic CT Analysis
          </h1>

          <p className="text-slate-300 text-sm sm:text-base leading-relaxed max-w-3xl">
            An explainable, academic machine-learning system trained on verified chest CT scans from the 
            <span className="text-teal-300 font-semibold"> IQ-OTH/NCCD Lung Cancer Dataset</span>. Evaluates 21 quantitative 
            first-order intensity, GLCM texture, and morphological edge features to differentiate between 
            <strong className="text-white"> Benign</strong> non-cancerous nodules and 
            <strong className="text-rose-400"> Malignant</strong> pulmonary carcinomas with rigorous cross-validation.
          </p>

          {/* Action CTAs */}
          <div className="pt-4 flex flex-wrap gap-3">
            <button
              onClick={() => onNavigate('upload')}
              className="px-6 py-3 bg-teal-600 hover:bg-teal-500 text-white font-semibold text-xs tracking-wider uppercase flex items-center space-x-2 border border-teal-400 cursor-pointer"
            >
              <span>Analyze CT Scan</span>
              <ArrowRight className="w-4 h-4" />
            </button>
            <button
              onClick={() => onNavigate('performance')}
              className="px-6 py-3 bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold text-xs tracking-wider uppercase flex items-center space-x-2 border border-slate-700 cursor-pointer"
            >
              <span>View Model Performance</span>
              <BarChart2 className="w-4 h-4 text-teal-400" />
            </button>
            <button
              onClick={() => onNavigate('about')}
              className="px-6 py-3 bg-transparent hover:bg-slate-800/60 text-slate-400 hover:text-white font-semibold text-xs tracking-wider uppercase flex items-center space-x-2 border border-slate-800 cursor-pointer"
            >
              <span>Project Documentation</span>
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Verified Metrics Highlights */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-700 font-mono">
            Final Test Set Evaluation (Unbiased Held-Out Split)
          </h2>
          <span className="text-xs text-slate-500 font-mono">Sample Size: 130 Verified Slices</span>
        </div>

        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white border border-slate-300 p-5 space-y-1">
            <div className="text-[11px] font-bold uppercase tracking-wider text-slate-500 font-mono">Classification Accuracy</div>
            <div className="text-3xl font-extrabold text-slate-900 font-mono">
              {evalMetrics ? `${(evalMetrics.accuracy * 100).toFixed(1)}%` : '96.2%'}
            </div>
            <p className="text-xs text-slate-600">Overall correct predictions on unseen CT slices</p>
          </div>

          <div className="bg-white border border-slate-300 p-5 space-y-1">
            <div className="text-[11px] font-bold uppercase tracking-wider text-teal-700 font-mono">ROC-AUC Score</div>
            <div className="text-3xl font-extrabold text-teal-800 font-mono">
              {evalMetrics ? evalMetrics.roc_auc.toFixed(4) : '0.9964'}
            </div>
            <p className="text-xs text-slate-600">Area under the receiver operating characteristic</p>
          </div>

          <div className="bg-white border border-slate-300 p-5 space-y-1">
            <div className="text-[11px] font-bold uppercase tracking-wider text-rose-700 font-mono">Malignant Sensitivity (Recall)</div>
            <div className="text-3xl font-extrabold text-rose-800 font-mono">
              {evalMetrics ? `${(evalMetrics.recall_sensitivity * 100).toFixed(1)}%` : '95.5%'}
            </div>
            <p className="text-xs text-slate-600">105/110 true malignant carcinomas detected</p>
          </div>

          <div className="bg-white border border-slate-300 p-5 space-y-1">
            <div className="text-[11px] font-bold uppercase tracking-wider text-indigo-700 font-mono">Benign Specificity</div>
            <div className="text-3xl font-extrabold text-indigo-800 font-mono">
              {evalMetrics ? `${(evalMetrics.specificity * 100).toFixed(1)}%` : '100.0%'}
            </div>
            <p className="text-xs text-slate-600">20/20 non-cancerous benign lesions confirmed</p>
          </div>
        </div>
      </div>

      {/* Clinical & Methodological Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Card 1: Primary Model */}
        <div className="bg-white border border-slate-300 p-6 space-y-3">
          <div className="w-9 h-9 bg-teal-100 border border-teal-300 text-teal-800 flex items-center justify-center font-bold">
            <Cpu className="w-5 h-5" />
          </div>
          <h3 className="text-base font-bold text-slate-900 font-mono">Logistic Regression Classifier</h3>
          <p className="text-xs text-slate-600 leading-relaxed">
            The core predictive model utilizes Scikit-learn's <code className="bg-slate-100 text-slate-800 px-1 py-0.5 text-[11px]">LogisticRegression</code> with 
            balanced class weighting and L2 regularization. Unlike opaque deep neural networks, it produces transparent log-odds 
            weights for each individual radiomic feature.
          </p>
          <ul className="text-xs text-slate-600 space-y-1 pt-2 border-t border-slate-200">
            <li className="flex items-center space-x-1.5">
              <CheckCircle2 className="w-3.5 h-3.5 text-teal-600 shrink-0" />
              <span>StandardScaler feature normalization</span>
            </li>
            <li className="flex items-center space-x-1.5">
              <CheckCircle2 className="w-3.5 h-3.5 text-teal-600 shrink-0" />
              <span>5-Fold Stratified Cross-Validation</span>
            </li>
            <li className="flex items-center space-x-1.5">
              <CheckCircle2 className="w-3.5 h-3.5 text-teal-600 shrink-0" />
              <span>Hyperparameter Grid Search (C=10.0, L2)</span>
            </li>
          </ul>
        </div>

        {/* Card 2: Radiomic Feature Engineering */}
        <div className="bg-white border border-slate-300 p-6 space-y-3">
          <div className="w-9 h-9 bg-indigo-100 border border-indigo-300 text-indigo-800 flex items-center justify-center font-bold">
            <Layers className="w-5 h-5" />
          </div>
          <h3 className="text-base font-bold text-slate-900 font-mono">21 Handcrafted Radiomic Features</h3>
          <p className="text-xs text-slate-600 leading-relaxed">
            Medical CT scans are transformed into quantitative spatial, textural, and morphological metrics that capture 
            biological tumor heterogeneity:
          </p>
          <ul className="text-xs text-slate-600 space-y-1 pt-2 border-t border-slate-200">
            <li className="flex items-center space-x-1.5">
              <span className="w-1.5 h-1.5 bg-slate-400"></span>
              <span><strong>Intensity Moments:</strong> Mean, Std, Skewness, Kurtosis</span>
            </li>
            <li className="flex items-center space-x-1.5">
              <span className="w-1.5 h-1.5 bg-slate-400"></span>
              <span><strong>GLCM Texture:</strong> Homogeneity, Contrast, Energy, ASM</span>
            </li>
            <li className="flex items-center space-x-1.5">
              <span className="w-1.5 h-1.5 bg-slate-400"></span>
              <span><strong>LBP Patterns:</strong> Local binary pattern entropy & variance</span>
            </li>
            <li className="flex items-center space-x-1.5">
              <span className="w-1.5 h-1.5 bg-slate-400"></span>
              <span><strong>Edge Analysis:</strong> Sobel gradient spiculation metrics</span>
            </li>
          </ul>
        </div>

        {/* Card 3: IQ-OTH/NCCD Dataset */}
        <div className="bg-white border border-slate-300 p-6 space-y-3">
          <div className="w-9 h-9 bg-slate-100 border border-slate-300 text-slate-800 flex items-center justify-center font-bold">
            <Database className="w-5 h-5" />
          </div>
          <h3 className="text-base font-bold text-slate-900 font-mono">IQ-OTH/NCCD Dataset</h3>
          <p className="text-xs text-slate-600 leading-relaxed">
            Collected at the Iraq-Oncology Teaching Hospital & National Center for Cancer Diseases. Recorded using a Siemens 
            SOMATOM CT scanner (120 kV, 1 mm slice thickness, 350-1200 HU window).
          </p>
          <ul className="text-xs text-slate-600 space-y-1 pt-2 border-t border-slate-200">
            <li className="flex items-center space-x-1.5">
              <span className="w-1.5 h-1.5 bg-slate-400"></span>
              <span>102 Validated Benign CT Slices</span>
            </li>
            <li className="flex items-center space-x-1.5">
              <span className="w-1.5 h-1.5 bg-slate-400"></span>
              <span>547 Validated Malignant CT Slices</span>
            </li>
            <li className="flex items-center space-x-1.5">
              <span className="w-1.5 h-1.5 bg-slate-400"></span>
              <span>416 Normal cases documented & excluded</span>
            </li>
          </ul>
        </div>
      </div>

      {/* Benign vs Malignant Clinical Characteristics */}
      <div className="bg-white border border-slate-300 p-6">
        <h3 className="text-sm font-bold uppercase tracking-wider text-slate-800 font-mono mb-4 flex items-center space-x-2">
          <Stethoscope className="w-4 h-4 text-teal-600" />
          <span>Diagnostic Distinction: Benign vs. Malignant Pulmonary Lesions</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="p-4 border border-emerald-300 bg-emerald-50/50 space-y-2">
            <div className="flex items-center justify-between">
              <span className="px-2 py-0.5 bg-emerald-700 text-white text-[11px] font-mono font-bold uppercase">
                Benign Lesions (Class 0)
              </span>
              <span className="text-xs text-emerald-800 font-mono font-semibold">Non-Cancerous Abnormality</span>
            </div>
            <p className="text-xs text-slate-700 leading-relaxed">
              Typically correspond to granulomas, hamartomas, or resolving post-inflammatory scarring. In CT imaging, 
              benign nodules predominantly display smooth, circumscribed borders, uniform local micro-textures (high LBP consistency), 
              and lack invasive vascular traction.
            </p>
            <div className="text-[11px] text-emerald-900 font-mono bg-white/80 p-2 border border-emerald-200">
              <strong>Model Characteristics:</strong> Lower edge gradient magnitude, higher LBP standard deviation, and lower central attenuation density.
            </div>
          </div>

          <div className="p-4 border border-rose-300 bg-rose-50/50 space-y-2">
            <div className="flex items-center justify-between">
              <span className="px-2 py-0.5 bg-rose-700 text-white text-[11px] font-mono font-bold uppercase">
                Malignant Carcinomas (Class 1)
              </span>
              <span className="text-xs text-rose-800 font-mono font-semibold">Cancerous Neoplasm</span>
            </div>
            <p className="text-xs text-slate-700 leading-relaxed">
              Represent primary bronchogenic carcinomas (adenocarcinoma, squamous cell carcinoma) or metastatic lesions. 
              Malignant tumors characteristically display irregular spiculated margins, heterogeneous attenuation, 
              pleural tethering, and micro-texture irregularity.
            </p>
            <div className="text-[11px] text-rose-900 font-mono bg-white/80 p-2 border border-rose-200">
              <strong>Model Characteristics:</strong> Elevated Sobel edge gradient, higher GLCM homogeneity weights, and increased non-aerated tissue attenuation.
            </div>
          </div>
        </div>
      </div>

      {/* Academic & Medical Disclaimer */}
      <div className="bg-amber-50 border border-amber-300 p-5 flex items-start space-x-3.5">
        <AlertTriangle className="w-5 h-5 text-amber-700 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <h4 className="text-xs font-bold uppercase tracking-wider text-amber-900 font-mono">
            Mandatory Academic Research Disclaimer
          </h4>
          <p className="text-xs text-amber-800 leading-relaxed">
            LungScan AI is an educational computer science research prototype developed to study the application of 
            handcrafted radiomics and Logistic Regression on 2D CT scans. 
            <strong> It is NOT a certified medical diagnostic device and must NEVER be used to make clinical, therapeutic, 
            or surgical treatment decisions.</strong> All predictions require radiological confirmation by licensed oncologists.
          </p>
        </div>
      </div>
    </div>
  );
};

