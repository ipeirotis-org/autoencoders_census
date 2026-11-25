/**
 * Detection method metadata served alongside model presets by
 * `GET /api/jobs/presets`.
 *
 * The Python `model/presets.py` module (`MODEL_TYPE_INFO` / `VALID_MODEL_TYPES`)
 * is the canonical source of truth. We mirror the frontend-facing metadata here
 * following the same pattern as `frontend/server/utils/modelPresets.ts`.
 *
 * If you add a detection method:
 *   1. Add it to `model/presets.py` (`MODEL_TYPE_INFO` and `VALID_MODEL_TYPES`).
 *   2. Add it to `VALID_MODEL_TYPES` in `frontend/server/middleware/validation.ts`.
 *   3. Add it to `MODEL_METHODS` below.
 *
 * `tests/test_model_presets.py` enforces a drift-guard comparing these ids
 * against Python `MODEL_TYPE_INFO`.
 */

export interface ModelMethodInfo {
  /** Stable detection method id sent to the worker via `modelType`. */
  id: 'chow_liu' | 'autoencoder';
  /** Short human-readable label for the UI selector. */
  label: string;
  /** Longer description shown to help researchers choose a method. */
  description: string;
}

export const MODEL_METHODS: ReadonlyArray<ModelMethodInfo> = [
  {
    id: 'chow_liu',
    label: 'Chow-Liu Tree',
    description:
      'Closed-form mutual-information tree. Deterministic, zero hyperparameters, and finishes in under a second. Recommended for most surveys.',
  },
  {
    id: 'autoencoder',
    label: 'Autoencoder (Neural)',
    description:
      'Neural autoencoder with percentile-trimmed categorical cross-entropy. Captures higher-order multivariate interactions; choose a model size preset below.',
  },
];
