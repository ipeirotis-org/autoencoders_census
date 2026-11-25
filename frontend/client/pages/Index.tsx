/**
 * Main UI - dropzone, preview, results display
 */

import React, { useState, useCallback, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { Dropzone } from "@/components/Dropzone";
import { PreviewTable } from "@/components/PreviewTable"; // We reuse this for results!
import { ResultCard } from "@/components/ResultCard";
import { parseCSVFile, type CSVParseResult } from "@/utils/csv-parser";
import {
  uploadCsv,
  checkJobStatus,
  fetchModelOptions,
  ApiError,
  type JobStatus,
  type ModelMethodInfo,
  type ModelPresetInfo,
} from "@/utils/api";
import { resolveJobError } from "@/utils/jobErrors";
import { logout as logoutRequest } from "@/utils/auth";
import { useCurrentUser } from "@/hooks/useCurrentUser";
import { cn } from "@/lib/utils";
import { PreviewErrorBoundary } from "@/components/error-boundaries/PreviewErrorBoundary";
import { ResultsErrorBoundary } from "@/components/error-boundaries/ResultsErrorBoundary";

// Fallback detection methods used when /api/jobs/presets fails (e.g. offline
// dev). Mirrors `frontend/server/utils/modelMethods.ts`.
const FALLBACK_METHODS: ModelMethodInfo[] = [
  {
    id: "chow_liu",
    label: "Chow-Liu Tree",
    description:
      "Closed-form mutual-information tree. Deterministic, zero hyperparameters, and finishes in under a second. Recommended for most surveys.",
  },
  {
    id: "autoencoder",
    label: "Autoencoder (Neural)",
    description:
      "Neural autoencoder with percentile-trimmed categorical cross-entropy. Captures higher-order multivariate interactions; choose a model size preset below.",
  },
];

// Fallback preset list used when the /api/jobs/presets fetch fails (e.g.
// offline dev). Mirrors the canonical list in
// `frontend/server/utils/modelPresets.ts` so the dropdown still renders.
const FALLBACK_PRESETS: ModelPresetInfo[] = [
  { id: "auto", label: "Auto", description: "Pick a preset automatically." },
  { id: "small", label: "Small", description: "Compact 1-layer model." },
  { id: "medium", label: "Medium", description: "Balanced 2-layer model." },
  { id: "large", label: "Large", description: "Higher-capacity 3-layer model." },
];

export default function Index() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<CSVParseResult | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<JobStatus["status"]>("uploading"); // Reusing for idle state
  const [results, setResults] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<any>(null);
  const [loggingOut, setLoggingOut] = useState(false);
  // Detection method selector ('chow_liu' or 'autoencoder') plus autoencoder
  // size preset dropdown (TASKS.md 3.2). Both lists hydrate from
  // /api/jobs/presets with local fallbacks.
  const [methods, setMethods] = useState<ModelMethodInfo[]>(FALLBACK_METHODS);
  const [selectedMethod, setSelectedMethod] = useState<string>("chow_liu");
  const [presets, setPresets] = useState<ModelPresetInfo[]>(FALLBACK_PRESETS);
  const [selectedPreset, setSelectedPreset] = useState<string>("auto");
  const { toast } = useToast();
  const navigate = useNavigate();
  const { user, setUser } = useCurrentUser();

  // Fetch detection methods and model presets once on mount without blocking
  // rendering (FALLBACK_METHODS / FALLBACK_PRESETS render immediately).
  useEffect(() => {
    let cancelled = false;
    fetchModelOptions().then(({ presets: fetchedPresets, methods: fetchedMethods }) => {
      if (cancelled) return;
      if (fetchedPresets.length > 0) {
        setPresets(fetchedPresets);
      }
      if (fetchedMethods.length > 0) {
        setMethods(fetchedMethods);
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const handleLogout = useCallback(async () => {
    setLoggingOut(true);
    try {
      await logoutRequest();
    } catch (err) {
      // Even if the server call fails (e.g. session already gone), clear
      // local state so the user is bounced back to the auth screen.
      console.error("Logout request failed", err);
    } finally {
      // Clearing the cached user flips <AuthGate> back to <AuthScreen>.
      setUser(null);
      setLoggingOut(false);
    }
  }, [setUser]);

  // Polling Logic
  useEffect(() => {
    if (!jobId || status === "complete" || status === "error") return;

    const interval = setInterval(async () => {
      try {
        const data = await checkJobStatus(jobId);
        if (data.status === "complete") {
          setResults(data.outliers || []);
          setStats(data.stats);
          setStatus("complete");
          toast({ title: "Analysis Complete", description: "Outliers detected successfully." });
          clearInterval(interval);
        } else if (data.status === "error") {
          // TASKS.md 2.3: prefer the structured {heading, message} from
          // resolveJobError so the UI always shows a clean, friendly
          // failure state even when the worker didn't provide a message
          // string (legacy error payloads, internal crashes, etc.).
          const { heading, message } = resolveJobError(data);
          setError(`${heading}: ${message}`);
          setStatus("error");
          clearInterval(interval);
        }
      } catch (e) {
        // If the session expired mid-poll, bounce the user back to the
        // auth screen so they can log in again instead of looping on 401s.
        if (e instanceof ApiError && e.status === 401) {
          clearInterval(interval);
          setUser(null);
          return;
        }
        console.error("Polling error", e);
      }
    }, 2000); // Check every 2 seconds

    return () => clearInterval(interval);
  }, [jobId, status, toast, setUser]);

  const handleFileSelect = useCallback(async (selectedFile: File) => {
    setFile(selectedFile);
    setError(null);
    setResults([]);
    setJobId(null);
    setStatus("uploading"); // Reset to initial state (idle)

    try {
      const parsed = await parseCSVFile(selectedFile);
      setPreview(parsed);
    } catch (err) {
      setError("Failed to parse CSV preview");
    }
  }, []);

  const handleUpload = useCallback(async () => {
    if (!file) return;
    setStatus("processing"); // Show spinner

    try {
      const presetToSend =
        selectedMethod === "autoencoder" ? selectedPreset : undefined;
      const response = await uploadCsv(file, presetToSend, selectedMethod);
      // Navigate to the dedicated progress page so the user gets the
      // full monitoring/cancel/export UI introduced in Phase 4.
      navigate(`/job/${response.jobId}`);
    } catch (err: any) {
      // Session expired between page load and upload - bounce back to auth.
      if (err instanceof ApiError && err.status === 401) {
        setUser(null);
        return;
      }
      setError(err.message);
      setStatus("error");
    }
  }, [file, selectedMethod, selectedPreset, navigate, setUser]);

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setJobId(null);
    setResults([]);
    setError(null);
    setStatus("uploading");
  };

  const getOrderedHeaders = (row: any) => {
    if (!row) return [];
    // Outlier records may nest user columns under `data` (Phase 4 format)
    // or keep them flat (legacy format). Handle both.
    const source = row.data && typeof row.data === 'object' ? row.data : row;
    const allHeaders = Object.keys(source);

    // Remove 'reconstruction_error' from the list
    const dataHeaders = allHeaders.filter(h => h !== 'reconstruction_error');

    // Put 'reconstruction_error' at the very start
    return ['reconstruction_error', ...dataHeaders];
  };


  const isProcessing = status === "processing" || (jobId && status !== "complete" && status !== "error");

  return (
    <div className="min-h-screen bg-slate-50 py-8 px-4">
      <div className="max-w-5xl mx-auto space-y-6">
        
        {/* Header */}
        <div className="flex items-center justify-between gap-4 mb-8">
          <div className="flex items-center gap-4">
            <img src="/AutoEncoder_logo_black.png" alt="Logo" className="w-16 h-16 object-contain" />
            <h1 className="text-4xl font-bold text-gray-900">Outlier Detection</h1>
          </div>
          {user && (
            <div className="flex items-center gap-3 text-sm">
              <span className="text-gray-600 hidden sm:inline" title={user.email}>
                {user.email}
              </span>
              <Button
                variant="outline"
                size="sm"
                onClick={handleLogout}
                disabled={loggingOut}
              >
                {loggingOut ? "Signing out..." : "Sign out"}
              </Button>
            </div>
          )}
        </div>

        {/* 1. Upload Section */}
        {!jobId && (
          <div className="bg-white rounded-2xl shadow p-6">
            <Dropzone onFileSelect={handleFileSelect} disabled={isProcessing} />
            
            {/* Input Preview */}
            {preview && (
              <div className="mt-6">
                <h3 className="text-sm font-semibold mb-2">Input Preview</h3>
                <PreviewErrorBoundary>
                  <PreviewTable rows={preview.rows} headers={preview.headers} totalRows={preview.totalRows} />
                </PreviewErrorBoundary>

                {/* Detection Method Picker: Chow-Liu Tree vs Neural Autoencoder */}
                <div className="mt-6">
                  <span
                    id="detection-method-label"
                    className="block text-sm font-semibold mb-2 text-gray-800"
                  >
                    Detection method
                  </span>
                  <div
                    role="radiogroup"
                    aria-labelledby="detection-method-label"
                    className="grid grid-cols-1 sm:grid-cols-2 gap-3"
                  >
                    {methods.map((method) => {
                      const isSelected = selectedMethod === method.id;
                      const badge =
                        method.id === "chow_liu"
                          ? "Fast · Deterministic"
                          : "Deep Learning";
                      return (
                        <button
                          key={method.id}
                          type="button"
                          role="radio"
                          aria-checked={isSelected}
                          disabled={isProcessing}
                          onClick={() => setSelectedMethod(method.id)}
                          className={cn(
                            "text-left rounded-lg border p-3.5 transition-all focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-60 disabled:cursor-not-allowed",
                            isSelected
                              ? "border-blue-600 bg-blue-50/60 ring-1 ring-blue-600"
                              : "border-gray-200 bg-white hover:border-gray-300"
                          )}
                        >
                          <div className="flex items-center justify-between gap-2">
                            <span className="text-sm font-semibold text-gray-900">
                              {method.label}
                            </span>
                            <span
                              className={cn(
                                "text-[11px] font-medium px-2 py-0.5 rounded-full",
                                isSelected
                                  ? "bg-blue-100 text-blue-800"
                                  : "bg-gray-100 text-gray-600"
                              )}
                            >
                              {badge}
                            </span>
                          </div>
                          <p className="mt-1.5 text-xs text-gray-600 leading-relaxed">
                            {method.description}
                          </p>
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* TASKS.md 3.2: Autoencoder size preset picker — shown only
                    when the neural autoencoder is selected (Chow-Liu is
                    closed-form with zero tuning hyperparameters). */}
                {selectedMethod === "autoencoder" && (
                  <div className="mt-4">
                    <label
                      htmlFor="model-preset"
                      className="block text-sm font-semibold mb-1 text-gray-800"
                    >
                      Autoencoder model preset
                    </label>
                    <select
                      id="model-preset"
                      value={selectedPreset}
                      onChange={(e) => setSelectedPreset(e.target.value)}
                      disabled={isProcessing}
                      className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
                    >
                      {presets.map((preset) => (
                        <option key={preset.id} value={preset.id}>
                          {preset.label}
                        </option>
                      ))}
                    </select>
                    <p className="mt-1 text-xs text-gray-500">
                      {presets.find((p) => p.id === selectedPreset)?.description ||
                        "Choose a model size for the autoencoder."}
                    </p>
                  </div>
                )}

                <div className="mt-4 flex gap-4">
                  <Button onClick={handleUpload} disabled={isProcessing} size="lg" className="flex-1">
                    {isProcessing ? "Processing..." : "Run Analysis"}
                  </Button>
                  <Button onClick={handleReset} variant="outline" size="lg">Reset</Button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* 2. Processing State */}
        {isProcessing && (
          <div className="bg-white rounded-2xl shadow p-12 text-center">
            <div className="animate-spin w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full mx-auto mb-4"></div>
            <h2 className="text-xl font-semibold">Analyzing Data...</h2>
            <p className="text-gray-500">This may take a minute. Detecting outliers.</p>
          </div>
        )}

        {/* 3. Results Section */}
        {status === "complete" && results.length > 0 && (
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            
            {/* Left Column: Outlier Table (Takes 75% width) */}
            <div className="lg:col-span-3 bg-white rounded-2xl shadow p-6">
            <div className="flex justify-between items-center mb-6">
              <div>
                <h2 className="text-2xl font-bold text-green-700">Analysis Complete</h2>
                <p className="text-gray-600">
                  Found {results.length} outliers. 
                  Kept {stats?.kept_columns?.length || 0} columns.
                </p>
              </div>
              <Button onClick={handleReset} variant="outline">Analyze New File</Button>
            </div>

            <ResultsErrorBoundary>
              <ResultCard type="success" message="Outliers identified successfully." />
            </ResultsErrorBoundary>

            <div className="mt-6">
              <h3 className="text-lg font-semibold mb-3">Top Outliers</h3>

              {/* [STEP 2] Update the headers prop here */}
              <PreviewErrorBoundary>
                <PreviewTable
                  rows={results.map((r: any) => {
                    // Flatten nested data/metadata structure for display.
                    // Phase 4 outlier records nest user columns under `data`;
                    // legacy records keep them flat. Merge both into a flat
                    // object so PreviewTable can render column values directly.
                    if (r.data && typeof r.data === 'object') {
                      const { data, contributions, ...meta } = r;
                      return { ...data, ...meta };
                    }
                    return r;
                  })}
                  headers={getOrderedHeaders(results[0])}
                  totalRows={results.length}
                />
              </PreviewErrorBoundary>

            </div>
          </div>

            {/* Right Column: Dropped Columns Panel (Takes 25% width) */}
            <div className="lg:col-span-1 bg-white rounded-2xl shadow p-6 h-fit max-h-[80vh] flex flex-col">
              <div className="mb-4">
                <h3 className="font-bold text-gray-800 text-lg">Dropped Columns</h3>
                <p className="text-xs text-gray-500">
                  Removed due to high cardinality ({'>'}9) or being a single value.
                </p>
              </div>
              
              <div className="overflow-y-auto pr-2 space-y-2 flex-1">
                {stats?.ignored_columns?.length === 0 ? (
                  <p className="text-sm text-gray-400 italic">No columns dropped.</p>
                ) : (
                  stats?.ignored_columns?.map((col: any, idx: number) => (
                    <div key={idx} className="p-3 bg-slate-50 rounded border border-slate-200 text-sm group hover:border-red-300 transition-colors">
                      <div className="font-semibold text-slate-700 truncate" title={col.name}>
                        {col.name}
                      </div>
                      <div className="flex justify-between items-center mt-1">
                        <span className="text-xs text-slate-500">Unique Values:</span>
                        <span className="text-xs font-mono bg-slate-200 px-1.5 py-0.5 rounded text-slate-600">
                          {col.unique_values}
                        </span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="bg-white rounded-2xl shadow p-6">
             <ResultsErrorBoundary>
               <ResultCard type="error" message={error} />
             </ResultsErrorBoundary>
             <Button onClick={handleReset} variant="outline" className="mt-4">Try Again</Button>
          </div>
        )}

      </div>
    </div>
  );
}
