export type NavPage = 'recognition' | 'dataset' | 'performance';

export interface PerClassMetrics {
  precision: number[];
  recall: number[];
  f1: number[];
  tp: number[];
  fp: number[];
  fn: number[];
  support: number[];
}

export interface ModelMetadata {
  n_samples: number;
  n_classes: number;
  image_shape: number[];
  intensity_range: number[];
  n_train: number;
  n_train_augmented?: number;
  n_test: number;
  n_correct: number;
  n_errors: number;
  model_name: string;
  kernel: string;
  C: number;
  gamma: number | string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  confusion_matrix: number[][];
  row_totals: number[];
  col_totals: number[];
  per_class: PerClassMetrics;
}

export interface DatasetSample {
  label: number;
  image: number[][];
  flat: number[];
}

export interface DatasetInfo {
  metadata: ModelMetadata;
  class_distribution: Record<string, number>;
  samples: DatasetSample[];
}

export interface PredictResult {
  predicted_digit: number;
  decision_scores: number[];
  preprocessed_pixels: number[];
}
