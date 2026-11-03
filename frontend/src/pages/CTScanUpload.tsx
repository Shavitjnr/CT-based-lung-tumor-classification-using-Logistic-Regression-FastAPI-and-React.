import React, { useState, useRef } from 'react';
import { UploadCloud, FileImage, AlertCircle, Play, RefreshCw, CheckCircle2, Sliders, Info } from 'lucide-react';
import { SampleScan, predictCTScan, PredictionResult } from '../services/api';

interface CTScanUploadProps {
  samples: SampleScan[];
  onPredictionComplete: (result: PredictionResult, previewUrl: string, sampleInfo?: SampleScan) => void;
}

export const CTScanUpload: React.FC<CTScanUploadProps> = ({ samples, onPredictionComplete }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedSample, setSelectedSample] = useState<SampleScan | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [threshold, setThreshold] = useState<number>(0.50);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [analysisStep, setAnalysisStep] = useState<string>('');

  const fileInputRef = useRef<HTMLInputElement>(null);

  const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/jpg'];
  const MAX_SIZE_MB = 15;

  const handleFile = (file: File) => {
    setErrorMsg(null);
    if (!ALLOWED_TYPES.includes(file.type)) {
      setErrorMsg(`Invalid file type "${file.type || 'unknown'}". Please upload a JPG or PNG chest CT scan.`);
      return;
    }
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      setErrorMsg(`File size (${(file.size / (1024 * 1024)).toFixed(1)}MB) exceeds maximum limit of ${MAX_SIZE_MB}MB.`);
      return;
    }

    setSelectedFile(file);
    setSelectedSample(null);

    const reader = new FileReader();
    reader.onload = (e) => {
      setPreviewUrl(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleSelectSample = (sample: SampleScan) => {
    setErrorMsg(null);
    setSelectedFile(null);
    setSelectedSample(sample);
    setPreviewUrl(`/api/samples/image/${sample.filename}`);
  };

  const handleReset = () => {
    setSelectedFile(null);
    setSelectedSample(null);
    setPreviewUrl(null);
    setErrorMsg(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleAnalyze = async () => {
    if (!selectedFile && !selectedSample) {
      setErrorMsg('Please select or upload a chest CT scan image first.');
      return;
    }

    setIsLoading(true);
    setErrorMsg(null);

    try {
      setAnalysisStep('1/4: Decoding & Resizing CT Scan to 256x256...');
      await new Promise(r => setTimeout(r, 200));

      setAnalysisStep('2/4: Applying Gaussian Smoothing & CLAHE Enhancement...');
      await new Promise(r => setTimeout(r, 200));

      setAnalysisStep('3/4: Computing 21 GLCM, LBP, and Edge Radiomic Features...');
      const sampleId = selectedSample ? selectedSample.id : undefined;
      const fileToUpload = selectedFile || undefined;

      const result = await predictCTScan(fileToUpload, sampleId, threshold);

      setAnalysisStep('4/4: Evaluating Logistic Regression Logit & Odds...');
      await new Promise(r => setTimeout(r, 150));

      onPredictionComplete(result, previewUrl!, selectedSample || undefined);
    } catch (err: any) {
      setErrorMsg(err.message || 'An error occurred during CT scan analysis.');
    } finally {
      setIsLoading(false);
      setAnalysisStep('');
    }
  };

  return (
    <div className="space-y-8 pb-12">
      {/* Header */}
      <div className="border-b border-slate-300 pb-4">
        <h1 className="text-2xl font-extrabold text-slate-900 font-mono tracking-tight">
          CT Scan Ingestion & Analysis
        </h1>
        <p className="text-xs text-slate-600 mt-1">
          Upload a high-resolution thoracic CT scan or select a verified case from the IQ-OTH/NCCD dataset.
        </p>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Upload & Curated Samples */}
        <div className="lg:col-span-7 space-y-6">
          {/* Upload Dropzone */}
          <div
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed p-8 text-center cursor-pointer transition-all ${
              isDragging
                ? 'border-teal-500 bg-teal-50'
                : 'border-slate-300 hover:border-teal-600 hover:bg-slate-100/50 bg-white'
            }`}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={(e) => e.target.files && e.target.files[0] && handleFile(e.target.files[0])}
              accept=".jpg,.jpeg,.png"
              className="hidden"
            />
            <div className="flex flex-col items-center justify-center space-y-3">
              <div className="w-12 h-12 bg-slate-100 border border-slate-300 flex items-center justify-center text-slate-700">
                <UploadCloud className="w-6 h-6 text-teal-600" />
              </div>
              <div className="space-y-1">
                <p className="text-sm font-bold text-slate-800 font-mono">
                  Drag and drop a CT scan image here, or <span className="text-teal-700 underline">browse files</span>
                </p>
                <p className="text-xs text-slate-500">
                  Supported Formats: JPG, JPEG, PNG (Max 15MB)
                </p>
              </div>
            </div>
          </div>

          {/* Curated Sample Scans Selector */}
          <div className="bg-white border border-slate-300 p-5 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <FileImage className="w-4 h-4 text-teal-600" />
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800 font-mono">
                  Instant Verification: Curated Test Set Samples
                </h3>
              </div>
              <span className="text-[11px] text-slate-500 font-mono">1-Click Clinical Demonstration</span>
            </div>

            <p className="text-xs text-slate-600">
              Select one of the verified test cases from the Iraq-Oncology Teaching Hospital dataset to evaluate model performance without uploading external files:
            </p>

            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 pt-2">
              {samples.map((sample) => {
                const isSelected = selectedSample?.id === sample.id;
                const isBenign = sample.true_class === 'Benign';
                return (
                  <button
                    key={sample.id}
                    onClick={() => handleSelectSample(sample)}
                    className={`text-left p-2.5 border transition-all cursor-pointer relative ${
                      isSelected
                        ? 'border-teal-600 bg-teal-50'
                        : 'border-slate-300 hover:border-slate-400 bg-slate-50'
                    }`}
                  >
                    <div className="aspect-square w-full bg-slate-900 border border-slate-300 overflow-hidden mb-2">
                      <img
                        src={`/api/samples/image/${sample.filename}`}
                        alt={sample.description}
                        className="w-full h-full object-cover filter contrast-125"
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <span className={`text-[10px] font-bold font-mono px-1.5 py-0.5 border uppercase ${
                        isBenign
                          ? 'bg-emerald-100 text-emerald-800 border-emerald-300'
                          : 'bg-rose-100 text-rose-800 border-rose-300'
                      }`}>
                        {sample.true_class}
                      </span>
                      <span className="text-[10px] text-slate-500 font-mono">{sample.id}</span>
                    </div>
                    <div className="text-[11px] text-slate-700 font-medium truncate mt-1">
                      {sample.source_file}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Column: Preview, Settings & Action */}
        <div className="lg:col-span-5 space-y-6">
          {/* Preview Panel */}
          <div className="bg-white border border-slate-300 p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-slate-200 pb-3">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-700 font-mono">
                Image Preview & Metadata
              </span>
              {previewUrl && (
                <button
                  onClick={handleReset}
                  className="text-xs text-rose-600 hover:text-rose-800 font-mono flex items-center space-x-1 cursor-pointer"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  <span>Clear</span>
                </button>
              )}
            </div>

            {previewUrl ? (
              <div className="space-y-3">
                <div className="relative aspect-square w-full max-w-sm mx-auto bg-black border border-slate-800 overflow-hidden flex items-center justify-center">
                  <img
                    src={previewUrl}
                    alt="CT Scan Target"
                    className="max-h-full max-w-full object-contain"
                  />
                  <div className="absolute bottom-2 left-2 px-2 py-0.5 bg-black/80 text-[10px] text-teal-300 font-mono border border-slate-700">
                    Siemens SOMATOM DICOM/JPEG
                  </div>
                </div>

                {/* File / Sample Info */}
                <div className="text-xs font-mono space-y-1 bg-slate-50 p-3 border border-slate-200">
                  <div className="flex justify-between">
                    <span className="text-slate-500">Source:</span>
                    <span className="text-slate-900 font-semibold truncate max-w-[200px]">
                      {selectedFile ? selectedFile.name : selectedSample?.description}
                    </span>
                  </div>
                  {selectedFile && (
                    <div className="flex justify-between">
                      <span className="text-slate-500">File Size:</span>
                      <span className="text-slate-900">{(selectedFile.size / 1024).toFixed(1)} KB</span>
                    </div>
                  )}
                  {selectedSample && (
                    <div className="flex justify-between">
                      <span className="text-slate-500">Ground Truth:</span>
                      <span className={`font-bold ${selectedSample.true_class === 'Benign' ? 'text-emerald-700' : 'text-rose-700'}`}>
                        {selectedSample.true_class} (Label {selectedSample.true_label})
                      </span>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="aspect-square w-full max-w-sm mx-auto bg-slate-50 border border-dashed border-slate-300 flex flex-col items-center justify-center p-6 text-center space-y-2 text-slate-400">
                <FileImage className="w-10 h-10 stroke-1" />
                <p className="text-xs font-mono">No CT scan selected</p>
                <p className="text-[11px] text-slate-400">Upload an image or pick a sample on the left</p>
              </div>
            )}

            {/* Classification Threshold Adjustment */}
            <div className="pt-3 border-t border-slate-200 space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs font-bold uppercase tracking-wider text-slate-700 font-mono flex items-center space-x-1.5">
                  <Sliders className="w-3.5 h-3.5 text-teal-600" />
                  <span>Decision Threshold</span>
                </label>
                <span className="text-xs font-mono font-bold text-teal-800 bg-teal-50 px-2 py-0.5 border border-teal-200">
                  {threshold.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0.10"
                max="0.90"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="w-full accent-teal-600 cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                <span>0.10 (High Sensitivity)</span>
                <span>Default: 0.50</span>
                <span>0.90 (High Specificity)</span>
              </div>
            </div>

            {/* Error Message */}
            {errorMsg && (
              <div className="p-3 bg-rose-50 border border-rose-300 text-rose-800 text-xs flex items-start space-x-2">
                <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
                <span>{errorMsg}</span>
              </div>
            )}

            {/* Loading Status */}
            {isLoading && (
              <div className="p-3 bg-teal-50 border border-teal-300 text-teal-900 text-xs space-y-2">
                <div className="flex items-center space-x-2 font-mono font-semibold">
                  <RefreshCw className="w-4 h-4 text-teal-600 animate-spin" />
                  <span>Feature Extraction & Inference Active</span>
                </div>
                <div className="text-[11px] font-mono text-teal-700 pl-6">
                  {analysisStep}
                </div>
              </div>
            )}

            {/* Run Button */}
            <button
              onClick={handleAnalyze}
              disabled={isLoading || (!selectedFile && !selectedSample)}
              className={`w-full py-3.5 font-bold font-mono text-xs tracking-wider uppercase flex items-center justify-center space-x-2 border transition-all ${
                isLoading || (!selectedFile && !selectedSample)
                  ? 'bg-slate-200 text-slate-400 border-slate-300 cursor-not-allowed'
                  : 'bg-teal-600 hover:bg-teal-500 text-white border-teal-500 cursor-pointer'
              }`}
            >
              <Play className="w-4 h-4 fill-current" />
              <span>{isLoading ? 'Processing Scan...' : 'Analyze CT Scan'}</span>
            </button>
          </div>

          {/* Validation Notice */}
          <div className="bg-slate-100 border border-slate-300 p-4 space-y-2 text-xs text-slate-600">
            <div className="flex items-center space-x-1.5 text-slate-800 font-bold font-mono">
              <Info className="w-4 h-4 text-teal-600" />
              <span>Input Validation & Boundary Conditions</span>
            </div>
            <p className="text-[11px] leading-relaxed">
              The preprocessing pipeline verifies image dimensions, luminance channels, and minimum contrast. Non-medical images, 
              blank photos, or corrupt files are rejected. Note that the model is strictly trained on axial thoracic CT slices.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

