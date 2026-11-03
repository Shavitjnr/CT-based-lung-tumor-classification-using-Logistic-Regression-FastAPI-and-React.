/**
 * api.ts - LungScan AI Client API Services
 */

export interface PredictionResult {
  status: string;
  predicted_class: 'Benign' | 'Malignant';
  predicted_label: number;
  confidence: number;
  probabilities: {
    benign: number;
    malignant: number;
  };
  threshold_applied: number;
  raw_logit: number;
  model_version: string;
  training_date: string;
  extracted_features: Record<string, number>;
  feature_contributions: Array<{
    feature: string;
    raw_value: number;
    z_score: number;
    coefficient: number;
    logit_impact: number;
    direction: string;
    description: string;
  }>;
  top_contributions: Array<{
    feature: string;
    raw_value: number;
    z_score: number;
    coefficient: number;
    logit_impact: number;
    direction: string;
    description: string;
  }>;
  explanation: string;
  preprocessed_stats: {
    raw_shape: number[];
    target_size: number[];
    parenchyma_ratio: number;
  };
  disclaimer: string;
}

export interface ModelMetricsResponse {
  evaluation_metrics: {
    accuracy: number;
    precision: number;
    recall_sensitivity: number;
    specificity: number;
    f1_score: number;
    roc_auc: number;
    brier_score: number;
    cv_mean_roc_auc: number;
    cv_std_roc_auc: number;
    cv_mean_train_roc_auc: number;
  };
  confusion_matrix: {
    true_negatives: number;
    false_positives: number;
    false_negatives: number;
    true_positives: number;
    total_test: number;
    matrix: number[][];
  };
  roc_curve: Array<{ fpr: number; tpr: number; threshold: number }>;
  pr_curve: Array<{ recall: number; precision: number; threshold: number }>;
  hyperparameters: Record<string, any>;
}

export interface ModelInfoResponse {
  model_name: string;
  model_version: string;
  algorithm: string;
  training_date: string;
  dataset: {
    name: string;
    source_institution: string;
    modality: string;
    total_binary_images: number;
    benign_count: number;
    malignant_count: number;
    normal_count_excluded: number;
    duplicates_removed: number;
  };
  feature_names: string[];
  feature_importance: Array<{
    feature: string;
    coefficient: number;
    abs_weight: number;
    odds_ratio: number;
    direction: string;
    description: string;
  }>;
  clinical_disclaimer: string;
}

export interface SampleScan {
  id: string;
  filename: string;
  true_class: 'Benign' | 'Malignant';
  true_label: number;
  description: string;
  source_file: string;
}

const API_BASE = '/api';

export async function fetchHealth(): Promise<any> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('Failed to reach backend health endpoint');
  return res.json();
}

export async function fetchModelMetrics(): Promise<ModelMetricsResponse> {
  const res = await fetch(`${API_BASE}/model/metrics`);
  if (!res.ok) throw new Error('Failed to load model metrics');
  return res.json();
}

export async function fetchModelInfo(): Promise<ModelInfoResponse> {
  const res = await fetch(`${API_BASE}/model/info`);
  if (!res.ok) throw new Error('Failed to load model info');
  return res.json();
}

export async function fetchSampleScans(): Promise<SampleScan[]> {
  const res = await fetch(`${API_BASE}/samples`);
  if (!res.ok) throw new Error('Failed to load sample scans');
  return res.json();
}

export async function predictCTScan(
  file?: File,
  sampleId?: string,
  threshold: number = 0.50
): Promise<PredictionResult> {
  const formData = new FormData();
  if (file) {
    formData.append('file', file);
  }
  if (sampleId) {
    formData.append('sample_id', sampleId);
  }
  formData.append('threshold', threshold.toString());

  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Prediction failed');
  }

  return res.json();
}

