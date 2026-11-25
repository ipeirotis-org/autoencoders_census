/**
 * Frontend API client - calls Express endpoints
 *
 * This orchestrates the 3-step upload:
 * 1. getUploadUrl()  → POST /api/jobs/upload-url  → Gets signed GCS URL
 * 2. uploadToGcs()   → PUT to signed URL          → Uploads file directly to GCS
 * 3. startJob()      → POST /api/jobs/start-job   → Triggers Pub/Sub message
 *
 * All credentialed requests include the session cookie via
 * `credentials: 'include'` so they work both same-origin and on cross-site
 * deployments where the SPA and API live on different hosts.
 */
export interface DemoResponse {
  message: string;
}

export interface UploadResponse {
  jobId: string;
}

export interface JobStatus {
  // "uploading" is a client-only idle state used by Index.tsx. The server
  // reports the JobStatus enum values from worker.py (queued, processing,
  // training, scoring, complete, error, canceled).
  status:
    | "uploading"
    | "queued"
    | "processing"
    | "training"
    | "scoring"
    | "complete"
    | "error"
    | "canceled";
  stats?: any;
  outliers?: any[];
  // TASKS.md 2.3: structured error fields written by worker.mark_job_error.
  // `error` is the human-readable message; `errorCode` is a stable
  // machine-readable identifier (see utils/jobErrors.ts); `errorType`
  // buckets the code into a pipeline stage.
  error?: string;
  errorCode?: string;
  errorType?: string;
  // Detection method chosen by the user ('autoencoder' or 'chow_liu').
  modelType?: string;
  // TASKS.md 3.2: model preset metadata. `modelPreset` is the resolved
  // preset id (small/medium/large) the worker actually used;
  // `modelPresetRequested` is what the user / dropdown asked for
  // (small/medium/large/auto). The two differ when the user picked
  // 'auto' and the worker auto-selected a concrete preset based on
  // dataset shape.
  modelPreset?: string;
  modelPresetRequested?: string;
}

/**
 * TASKS.md 3.2: Model preset metadata served by `GET /api/jobs/presets`.
 * Used to populate the upload-time dropdown. Mirrors the shape returned
 * by the Python `model.presets.list_presets()` helper.
 */
export interface ModelPresetInfo {
  id: string;
  label: string;
  description: string;
}

/**
 * Detection method metadata served alongside presets by `GET /api/jobs/presets`.
 * Mirrors `model.presets.list_model_types()` in Python.
 */
export interface ModelMethodInfo {
  id: string;
  label: string;
  description: string;
}

export interface ModelOptionsResponse {
  presets: ModelPresetInfo[];
  methods: ModelMethodInfo[];
}

/**
 * Thrown when a job API request fails. Carries the HTTP status so callers
 * can distinguish authentication failures (401, session expired - the user
 * needs to log back in) from validation errors and from real server errors.
 */
export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

async function parseError(res: Response, fallback: string): Promise<string> {
  try {
    const body = await res.json();
    if (body?.details?.[0]?.message) return body.details[0].message;
    if (typeof body?.error === "string") return body.error;
  } catch {
    // non-JSON error body, fall through
  }
  return fallback;
}

// 1. Get the Signed URL
async function getUploadUrl(filename: string, contentType: string) {
  const res = await fetch(`${API_BASE}/api/jobs/upload-url`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ filename, contentType }),
  });
  if (!res.ok) {
    throw new ApiError(await parseError(res, "Failed to get upload URL"), res.status);
  }
  return res.json();
}

// 2. Upload File to GCS
// The Content-Type used for the PUT MUST match whatever the server used
// when signing the URL - otherwise GCS rejects the upload. The server
// echoes nothing back; instead it signs using exactly the contentType
// we sent in step 1 (or omits the constraint entirely if we sent an
// empty one). We pass the same `file.type` through here for both calls,
// so the PUT header always matches the signing value.
async function uploadToGcs(url: string, file: File) {
  const res = await fetch(url, {
    method: "PUT",
    body: file,
    headers: { "Content-Type": file.type },
  });
  if (!res.ok) throw new Error("Failed to upload file to storage");
}

// 3. Notify Backend to Start Worker
async function startJob(
  jobId: string,
  gcsFileName: string,
  modelPreset?: string,
  modelType?: string
) {
  const body: Record<string, string> = { jobId, gcsFileName };
  if (modelType) {
    body.modelType = modelType;
  }
  if (modelPreset && modelType !== "chow_liu") {
    body.modelPreset = modelPreset;
  }
  const res = await fetch(`${API_BASE}/api/jobs/start-job`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new ApiError(await parseError(res, "Failed to start processing job"), res.status);
  }
}

// Main Upload Function
//
// `modelPreset` is the optional autoencoder preset id chosen via the dropdown
// (TASKS.md 3.2); `modelType` selects the detection method ('chow_liu' or
// 'autoencoder'). When omitted, the server defaults to 'autoencoder' / 'auto'.
export async function uploadCsv(
  file: File,
  modelPreset?: string,
  modelType?: string
): Promise<UploadResponse> {
  // A. Get URL - pass file.type so the server signs for that exact
  //    Content-Type (or omits the constraint if file.type is empty).
  const { url, jobId, gcsFileName } = await getUploadUrl(file.name, file.type);

  // B. Upload (sends the same file.type header)
  await uploadToGcs(url, file);

  // C. Start Job
  await startJob(jobId, gcsFileName, modelPreset, modelType);

  return { jobId };
}

/**
 * Fetch available detection methods and autoencoder presets from the API.
 * Returns empty arrays on failure so the UI can fall back cleanly without
 * blocking the upload workflow.
 */
export async function fetchModelOptions(): Promise<ModelOptionsResponse> {
  try {
    const res = await fetch(`${API_BASE}/api/jobs/presets`, {
      credentials: "include",
    });
    if (!res.ok) {
      return { presets: [], methods: [] };
    }
    const data = await res.json();
    return {
      presets: Array.isArray(data?.presets) ? (data.presets as ModelPresetInfo[]) : [],
      methods: Array.isArray(data?.methods) ? (data.methods as ModelMethodInfo[]) : [],
    };
  } catch {
    return { presets: [], methods: [] };
  }
}

/**
 * Fetch the available model presets from the API. Kept for backwards
 * compatibility with callers that only need presets.
 */
export async function fetchModelPresets(): Promise<ModelPresetInfo[]> {
  const { presets } = await fetchModelOptions();
  return presets;
}

// Poll Status
export async function checkJobStatus(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${API_BASE}/api/jobs/job-status/${jobId}`, {
    credentials: "include",
  });
  if (!res.ok) {
    throw new ApiError(await parseError(res, "Failed to check job status"), res.status);
  }
  return res.json();
}
