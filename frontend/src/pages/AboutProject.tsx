import React from 'react';
import { 
  Cpu, 
  Layers, 
  Database, 
  ShieldAlert, 
  ArrowRight, 
  FileCode, 
  GitBranch, 
  CheckCircle2, 
  Sliders,
  Scale
} from 'lucide-react';

export const AboutProject: React.FC = () => {
  return (
    <div className="space-y-10 pb-16">
      {/* Page Title */}
      <div className="border-b border-slate-300 pb-4">
        <h1 className="text-2xl font-extrabold text-slate-900 font-mono tracking-tight">
          Project Architecture & Scientific Documentation
        </h1>
        <p className="text-xs text-slate-600 mt-1">
          Detailed technical report on radiomic feature extraction, Logistic Regression modeling, and clinical evaluation.
        </p>
      </div>

      {/* Interactive Workflow Diagram */}
      <div className="bg-white border border-slate-300 p-6 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-200 pb-3">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800 font-mono">
            End-to-End Machine Learning Processing Pipeline
          </h3>
          <span className="text-[11px] font-mono text-teal-700 bg-teal-50 px-2 py-0.5 border border-teal-200">
            Reproducible Pipeline
          </span>
        </div>

        {/* Step-by-Step Flowchart */}
        <div className="grid grid-cols-1 md:grid-cols-6 gap-3 pt-2">
          {/* Step 1 */}
          <div className="p-3.5 bg-slate-50 border border-slate-300 space-y-2 relative">
            <div className="w-6 h-6 bg-slate-900 text-white font-mono text-xs flex items-center justify-center font-bold">1</div>
            <h4 className="text-xs font-bold text-slate-900 font-mono">CT Scan Ingestion</h4>
            <p className="text-[11px] text-slate-600 leading-snug">
              2D axial thoracic CT slice in JPG/PNG format (Siemens SOMATOM DICOM export).
            </p>
          </div>

          {/* Step 2 */}
          <div className="p-3.5 bg-slate-50 border border-slate-300 space-y-2">
            <div className="w-6 h-6 bg-teal-800 text-white font-mono text-xs flex items-center justify-center font-bold">2</div>
            <h4 className="text-xs font-bold text-slate-900 font-mono">Image Preprocessing</h4>
            <p className="text-[11px] text-slate-600 leading-snug">
              Grayscale conversion, uniform 256×256 resampling, Gaussian noise reduction, CLAHE contrast enhancement.
            </p>
          </div>

          {/* Step 3 */}
          <div className="p-3.5 bg-slate-50 border border-slate-300 space-y-2">
            <div className="w-6 h-6 bg-indigo-800 text-white font-mono text-xs flex items-center justify-center font-bold">3</div>
            <h4 className="text-xs font-bold text-slate-900 font-mono">Radiomic Features</h4>
            <p className="text-[11px] text-slate-600 leading-snug">
              Extraction of 21 statistical moments, GLCM co-occurrence matrix, LBP micro-patterns, Sobel edge gradients.
            </p>
          </div>

          {/* Step 4 */}
          <div className="p-3.5 bg-slate-50 border border-slate-300 space-y-2">
            <div className="w-6 h-6 bg-slate-800 text-white font-mono text-xs flex items-center justify-center font-bold">4</div>
            <h4 className="text-xs font-bold text-slate-900 font-mono">Feature Scaling</h4>
            <p className="text-[11px] text-slate-600 leading-snug">
              StandardScaler Z-score transformation (zero mean, unit variance) fitted strictly on training data.
            </p>
          </div>

          {/* Step 5 */}
          <div className="p-3.5 bg-slate-50 border border-slate-300 space-y-2">
            <div className="w-6 h-6 bg-teal-700 text-white font-mono text-xs flex items-center justify-center font-bold">5</div>
            <h4 className="text-xs font-bold text-slate-900 font-mono">Logistic Regression</h4>
            <p className="text-[11px] text-slate-600 leading-snug">
              Balanced class weighting, L2 regularization (C=10.0), computing the logit log-odds dot product.
            </p>
          </div>

          {/* Step 6 */}
          <div className="p-3.5 bg-slate-900 text-white border border-slate-800 space-y-2">
            <div className="w-6 h-6 bg-teal-400 text-slate-950 font-mono text-xs flex items-center justify-center font-bold">6</div>
            <h4 className="text-xs font-bold text-white font-mono">Classification</h4>
            <p className="text-[11px] text-slate-300 leading-snug">
              Sigmoid mapping to [0, 1] probability: Benign (0) vs. Malignant (1) + feature impact explanation.
            </p>
          </div>
        </div>
      </div>

      {/* Mathematical Rationale & Explainability */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white border border-slate-300 p-6 space-y-3">
          <div className="flex items-center space-x-2">
            <Scale className="w-5 h-5 text-teal-600" />
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-900 font-mono">
              Mathematical Formulation of Logistic Regression
            </h3>
          </div>

          <p className="text-xs text-slate-600 leading-relaxed">
            The classifier models the posterior probability of malignancy given a 21-dimensional normalized radiomic vector 
            <span className="font-mono font-semibold"> x ∈ ℝ²¹</span> using the standard logistic sigmoid function:
          </p>

          <div className="bg-slate-900 text-teal-300 p-4 font-mono text-xs space-y-1.5 border border-slate-800">
            <div>P(Y = 1 | x) = 1 / (1 + e^-(β₀ + ∑ βᵢ · xᵢ))</div>
            <div className="text-slate-400 text-[11px]">logit = β₀ + β₁·LBP_std + β₂·GLCM_homo + ... + β₂₁·Edge_density</div>
          </div>

          <p className="text-xs text-slate-600 leading-relaxed">
            Because each feature weight <span className="font-mono font-semibold">βᵢ</span> represents the change in log-odds 
            per standard-deviation shift in the feature, the model is fully glass-box and verifiable by medical researchers.
          </p>
        </div>

        <div className="bg-white border border-slate-300 p-6 space-y-3">
          <div className="flex items-center space-x-2">
            <Database className="w-5 h-5 text-teal-600" />
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-900 font-mono">
              Dataset Provenance & Class Selection
            </h3>
          </div>

          <p className="text-xs text-slate-600 leading-relaxed">
            The dataset originates from the <strong>Iraq-Oncology Teaching Hospital</strong> and the 
            <strong> National Center for Cancer Diseases (IQ-OTH/NCCD)</strong>. Slices were marked and validated 
            by board-certified radiologists:
          </p>

          <ul className="text-xs text-slate-700 space-y-1.5 font-mono">
            <li>• Benign Cases: 102 verified non-cancerous lesion slices</li>
            <li>• Malignant Cases: 547 verified carcinoma slices</li>
            <li>• Normal Cases: 416 healthy scans (strictly excluded from binary tumor classification)</li>
            <li>• 32 duplicate CT slices identified and purged via MD5 hashing</li>
          </ul>

          <p className="text-xs text-slate-500 italic">
            Normal scans were deliberately excluded to prevent trivial background classification and ensure the model learns 
            subtle differences between non-cancerous benign lesions and malignant neoplasms.
          </p>
        </div>
      </div>

      {/* Dataset Limitations & False Negative Analysis */}
      <div className="bg-white border border-slate-300 p-6 space-y-4">
        <h3 className="text-sm font-bold uppercase tracking-wider text-slate-900 font-mono flex items-center space-x-2">
          <ShieldAlert className="w-4 h-4 text-rose-600" />
          <span>Clinical Limitations & Medical Image Analysis Realities</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-slate-600">
          <div className="p-3 bg-slate-50 border border-slate-200 space-y-1.5">
            <h5 className="font-bold font-mono text-slate-800">1. Whole-Image Handcrafted Radiomics</h5>
            <p className="text-[11px] leading-relaxed">
              Without accurate 3D nodule voxel segmentation masks, handcrafted features are extracted across the full 
              thoracic CT slice, meaning mediastinal fat, ribs, and pleural boundaries contribute to the feature vector.
            </p>
          </div>

          <div className="p-3 bg-slate-50 border border-slate-200 space-y-1.5">
            <h5 className="font-bold font-mono text-slate-800">2. Critical Oncology False Negatives</h5>
            <p className="text-[11px] leading-relaxed">
              In clinical radiology, false negatives (misclassifying cancer as benign) are far more dangerous than false positives. 
              The decision threshold can be lowered from 0.50 to 0.20 to maximize sensitivity when screening high-risk cohorts.
            </p>
          </div>

          <div className="p-3 bg-slate-50 border border-slate-200 space-y-1.5">
            <h5 className="font-bold font-mono text-slate-800">3. Class Imbalance Handling</h5>
            <p className="text-[11px] leading-relaxed">
              With 547 malignant vs 102 benign cases, class weighting (<code className="bg-slate-200 px-1 py-0.5">balanced</code>) was applied 
              during Logistic Regression training to penalize errors on the minority benign class proportionally.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

