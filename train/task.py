"""
Vertex AI Container Entry Point - Runs inside Docker on Vertex AI.

This script is triggered by worker.py and executes the full ML pipeline:
1. Downloads CSV from GCS bucket
2. Cleans data (fill NaN, apply "Rule of 9" filter)
3. Vectorizes categorical columns (one-hot encoding)
4. Builds and trains an autoencoder (15 epochs)
5. Calculates reconstruction error for each row
6. Saves top 100 outliers to Firestore

Usage (called by Vertex AI):
    python task.py --job-id=abc123 --bucket-name=autoencoder_data --file-path=uploads/abc123/data.csv
"""

import argparse
import logging
import os
import numpy as np
import pandas as pd
from google.cloud import storage, firestore
from dataset.loader import DataLoader, DEFAULT_MAX_UNIQUE_VALUES
from model.presets import (
    MODEL_TYPE_CHOW_LIU,
    build_model_config,
    normalize_model_type,
    normalize_preset_name,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Force the client to use YOUR specific project, not the internal Google one
db = firestore.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT", "autoencoders-census"))


def _resolve_max_unique_values(default=DEFAULT_MAX_UNIQUE_VALUES):
    """Read the Rule-of-N threshold from the ``MAX_UNIQUE_VALUES`` env
    var, falling back to the built-in default (TASKS.md 3.1).

    Vertex AI containers inherit environment variables from the
    submitting worker, so setting ``MAX_UNIQUE_VALUES`` on the job
    environment propagates to the training container without code
    changes.
    """
    raw = os.getenv("MAX_UNIQUE_VALUES")
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            f"Ignoring non-integer MAX_UNIQUE_VALUES={raw!r}; "
            f"using default ({default})"
        )
        return default
    if value < 2:
        logger.warning(
            f"Ignoring MAX_UNIQUE_VALUES={value} (must be >= 2); "
            f"using default ({default})"
        )
        return default
    return value


def train_and_predict(
    job_id,
    bucket_name,
    file_path,
    max_unique_values=None,
    model_preset=None,
    model_type=None,
):
    """Run the end-to-end Vertex training pipeline.

    Args:
        job_id: Firestore job document ID.
        bucket_name: GCS bucket containing the uploaded CSV.
        file_path: GCS object path to the uploaded CSV.
        max_unique_values: Optional explicit Rule-of-N threshold. When
            ``None`` we fall back to the ``MAX_UNIQUE_VALUES``
            environment variable (via ``_resolve_max_unique_values``).
            The dispatcher in ``worker.process_upload_vertex`` passes
            this in as a CLI argument (``--max-unique-values=N``)
            because the Vertex training container does NOT inherit the
            dispatcher process's env vars — without explicit
            propagation the container silently defaulted to 9 even
            when the worker had ``MAX_UNIQUE_VALUES=15`` set,
            producing inconsistent feature filtering between local
            and Vertex modes (Codex P1 PR#49).
        model_preset: Optional preset id from :mod:`model.presets`
            (auto/small/medium/large). Forwarded by
            ``worker.process_upload_vertex`` via the
            ``--model-preset`` CLI flag (TASKS.md 3.2). Used only when
            ``model_type`` resolves to ``autoencoder``.
        model_type: Optional detection method id from :mod:`model.presets`
            (``autoencoder`` or ``chow_liu``). Forwarded by
            ``worker.process_upload_vertex`` via ``--model-type``.
            ``None`` and unknown values normalize to ``autoencoder``.
    """
    try:
        resolved_model_type = normalize_model_type(model_type)
        logger.info(
            f"Starting Vertex AI Job for {job_id} (model_type={resolved_model_type})"
        )

        # 1. Download Data
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        csv_bytes = blob.download_as_bytes()

        # 2. Load and Clean
        loader = DataLoader(drop_columns=[], rename_columns={}, columns_of_interest=[])
        df = loader.load_original_data(csv_bytes)

        # Rule of N Filter (TASKS.md 3.1: threshold is configurable via
        # the --max-unique-values CLI arg or MAX_UNIQUE_VALUES env var;
        # defaults to 9).
        if max_unique_values is None:
            max_unique_values = _resolve_max_unique_values()
        logger.info(f"Using Rule-of-N threshold: {max_unique_values}")
        stats = {"total_rows": len(df), "kept_columns": [], "ignored_columns": []}
        process_df = df.fillna("missing").astype(str)

        cols_to_keep = []
        for col in process_df.columns:
            unique_count = process_df[col].nunique()
            if 1 < unique_count <= max_unique_values:
                cols_to_keep.append(col)
                stats["kept_columns"].append({"name": col, "unique_values": unique_count})
            else:
                stats["ignored_columns"].append({"name": col, "unique_values": unique_count})

        process_df = process_df[cols_to_keep]

        if process_df.shape[1] == 0:
            raise ValueError(
                f"All columns were dropped! No columns fit the Rule of N "
                f"(N={max_unique_values})."
            )

        if resolved_model_type == MODEL_TYPE_CHOW_LIU:
            from chow_liu_rank import CLTree, per_column_contributions_from_nll

            logger.info("Fitting Chow-Liu tree...")
            cl_model = CLTree(alpha=1.0).fit(
                process_df, mi_subsample=10000, random_state=42
            )
            per_col_nll = cl_model.per_column_nll(process_df)
            anomaly_scores = per_col_nll.mean(axis=1)
            df["reconstruction_error"] = anomaly_scores

            top_outliers = (
                df.sort_values(by="reconstruction_error", ascending=False)
                .head(100)
                .copy()
            )
            top_positions = df.index.get_indexer(top_outliers.index)

            # Format "Original -> Predicted" using conditional MAP categories
            try:
                predicted_df = cl_model.predict_most_likely(
                    process_df.loc[top_outliers.index]
                )
                for col in predicted_df.columns:
                    if col in top_outliers.columns and col != "reconstruction_error":
                        original_vals = (
                            top_outliers[col].fillna("missing").astype(str)
                        )
                        predicted_vals = (
                            predicted_df[col].fillna("missing").astype(str)
                        )
                        top_outliers[col] = [
                            f"{orig} -> {pred}" if orig != pred else orig
                            for orig, pred in zip(original_vals, predicted_vals)
                        ]
            except Exception as decode_err:
                logger.error(f"Failed to format Chow-Liu predictions: {decode_err}")

            top_outliers = top_outliers.replace([np.inf, -np.inf], 0).fillna("missing")
            max_contributions = 10
            outliers_data = []
            for pos, (_, row) in zip(top_positions, top_outliers.iterrows()):
                contribs = per_column_contributions_from_nll(
                    per_col_nll[pos], cl_model.columns
                )[:max_contributions]
                row_dict = row.to_dict()
                raw_score = float(anomaly_scores[pos])
                score = raw_score if np.isfinite(raw_score) else 0.0
                row_dict["reconstruction_error"] = score
                outliers_data.append({
                    "data": row_dict,
                    "reconstruction_error": score,
                    "contributions": [
                        {"column": col, "percentage": float(pct)}
                        for col, pct in contribs
                    ],
                })

            db.collection('jobs').document(job_id).set({
                "status": "complete",
                "stats": stats,
                "outliers": outliers_data,
                "processedAt": firestore.SERVER_TIMESTAMP,
                "modelType": MODEL_TYPE_CHOW_LIU,
            }, merge=True)

            logger.info("Job Complete (chow_liu)")
            return

        # 3. Vectorization & model setup (split before fitting to prevent data leakage)
        from sklearn.model_selection import train_test_split as _tts
        from evaluate.outliers import compute_reconstruction_error
        from features.transform import Table2Vector
        from model.autoencoder import AutoencoderModel

        model_variable_types = {col: "categorical" for col in process_df.columns}
        vectorizer = Table2Vector(model_variable_types)

        train_df, test_df = _tts(process_df, test_size=0.2)
        vectorizer.fit(train_df)
        X_train = vectorizer.transform(train_df).astype('float32')
        X_test = vectorizer.transform(test_df).astype('float32')
        vectorized_df = vectorizer.transform(process_df).astype('float32')

        cardinalities = vectorizer.get_cardinalities(process_df.columns)
        logger.info("Initializing AutoEncoderModel...")
        ae_wrapper = AutoencoderModel(attribute_cardinalities=cardinalities)
        ae_wrapper.INPUT_SHAPE = X_train.shape[1:]

        # 4. Configure & Train (TASKS.md 3.2: pick a preset config from
        # the requested preset, or auto-select based on dataset shape).
        input_dim = X_train.shape[1]
        n_rows = len(process_df)
        model_config, resolved_preset = build_model_config(
            model_preset, input_dim=input_dim, n_rows=n_rows
        )
        requested_preset = normalize_preset_name(model_preset)
        logger.info(
            f"Vertex job {job_id} model preset: requested={requested_preset!r}, "
            f"resolved={resolved_preset!r}, input_dim={input_dim}, "
            f"n_rows={n_rows}"
        )

        # Pull training-loop knobs out before passing the rest to
        # build_autoencoder() (which only consumes network-shape keys).
        epochs = int(model_config.pop('epochs', 15))
        batch_size = int(model_config.pop('batch_size', 32))

        keras_model = ae_wrapper.build_autoencoder(model_config)

        logger.info(
            f"Training Model ({resolved_preset} preset, epochs={epochs}, "
            f"batch_size={batch_size})..."
        )
        keras_model.fit(
            X_train, X_train,
            epochs=epochs, batch_size=batch_size, verbose=2,
            validation_data=(X_test, X_test),
        )
        
        # 5. Predict & Score
        #
        # TASKS.md 2.8: Use the same per-attribute categorical crossentropy
        # (normalized by log(K)) as the CLI `find_outliers` command so that
        # web UI and CLI outlier rankings match on the same data/model.
        # Previously this path used raw MSE on one-hot vectors, which is a
        # fundamentally different scoring function and produced different
        # rankings than the CLI.
        reconstruction = keras_model.predict(vectorized_df)
        if isinstance(reconstruction, list):
            reconstruction = reconstruction[0]

        reconstruction_error = compute_reconstruction_error(
            vectorized_df, reconstruction, cardinalities
        )
        df['reconstruction_error'] = reconstruction_error

        top_outliers = df.sort_values(by='reconstruction_error', ascending=False).head(100)
        
        # --- NEW: Decode Reconstruction for Visualization ---
        # 1. Align reconstruction with dataframe indices
        reconstruction_df = pd.DataFrame(
            reconstruction, 
            columns=vectorized_df.columns, 
            index=df.index
        )
        
        # 2. Extract only the outliers' reconstruction
        # Use intersection to be safe (though indices should match)
        common_indices = top_outliers.index.intersection(reconstruction_df.index)
        outlier_reconstruction = reconstruction_df.loc[common_indices]
        
        # 3. Invert transformation to get categorical values.
        # Reset to contiguous 0..N index before tabularize so internal
        # concat doesn't misalign rows, then map back to common_indices.
        try:
            contiguous = outlier_reconstruction.reset_index(drop=True)
            decoded_outliers = vectorizer.tabularize_vector(contiguous)
            decoded_outliers.index = common_indices

            # 4. Format the `top_outliers` DataFrame to show "Original -> Predicted"
            for col in decoded_outliers.columns:
                if col in top_outliers.columns:
                    original_vals = top_outliers.loc[common_indices, col].fillna("missing").astype(str)
                    predicted_vals = decoded_outliers.loc[common_indices, col].fillna("missing").astype(str)
                    
                    # zip and format
                    formatted_col = [
                        f"{orig} -> {pred}" if orig != pred else orig
                        for orig, pred in zip(original_vals, predicted_vals)
                    ]
                    
                    top_outliers.loc[common_indices, col] = formatted_col
        except Exception as e:
            logger.error(f"Failed to decocde reconstruction: {e}")
            # Continue without formatting if deserialization fails
            pass
        # --- END NEW ---

        top_outliers = top_outliers.replace([np.inf, -np.inf], 0).fillna("missing")
        
        # 6. Save Results
        # TASKS.md 3.2: persist the resolved preset alongside the
        # results so the frontend can show "model: medium (auto)" on
        # the completed job. modelPresetRequested is also stored for
        # the auditing query "which presets did users actually pick?".
        db.collection('jobs').document(job_id).set({
            "status": "complete",
            "stats": stats,
            "outliers": top_outliers.to_dict(orient="records"),
            "processedAt": firestore.SERVER_TIMESTAMP,
            "modelType": resolved_model_type,
            "modelPreset": resolved_preset,
            "modelPresetRequested": requested_preset,
        }, merge=True)
        
        logger.info("Job Complete")
        
    except Exception as e:
        logger.error(f"Job failed: {e}")
        db.collection('jobs').document(job_id).set({"status": "error", "error": str(e)})
        
if __name__ == "__main__":
    # This part grabs the args passed by Vertex AI
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', type=str, required=True)
    parser.add_argument('--bucket-name', type=str, required=True)
    parser.add_argument('--file-path', type=str, required=True)
    # Rule-of-N threshold forwarded from worker.process_upload_vertex.
    # Optional: when absent we fall back to the MAX_UNIQUE_VALUES env
    # var, which in turn falls back to the module default (9). Vertex
    # AI CustomContainerTrainingJob.run() does not propagate the
    # dispatcher's env vars to the training container, so this CLI
    # arg is the only reliable channel for the override (Codex P1
    # PR#49).
    parser.add_argument(
        '--max-unique-values',
        type=int,
        default=None,
        dest='max_unique_values',
    )
    # Model preset id forwarded from worker.process_upload_vertex
    # (TASKS.md 3.2). Same env-var-vs-CLI-arg trade-off as
    # --max-unique-values: Vertex doesn't inherit dispatcher env vars,
    # so the worker forwards the user's choice via this flag. When
    # absent we pass None into train_and_predict, which is then
    # normalized to 'auto' inside build_model_config.
    parser.add_argument(
        '--model-preset',
        type=str,
        default=None,
        dest='model_preset',
    )
    # Detection method id forwarded from worker.process_upload_vertex
    # ('autoencoder' or 'chow_liu'). Absent defaults to 'autoencoder'.
    parser.add_argument(
        '--model-type',
        type=str,
        default=None,
        dest='model_type',
    )
    args = parser.parse_args()

    train_and_predict(
        args.job_id,
        args.bucket_name,
        args.file_path,
        max_unique_values=args.max_unique_values,
        model_preset=args.model_preset,
        model_type=args.model_type,
    )
