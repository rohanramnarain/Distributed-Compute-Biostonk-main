(function (global) {
  "use strict";

  function sigmoid(value) {
    return value >= 0 ? 1 / (1 + Math.exp(-value)) : Math.exp(value) / (1 + Math.exp(value));
  }

  function assertModel(model) {
    const dimension = Number(model?.embedding_dim);
    if (!Number.isInteger(dimension) || dimension < 1) throw new Error("The embedding model has an invalid dimension.");
    if (!Array.isArray(model.weights) || model.weights.length !== dimension) throw new Error("The embedding model weights do not match its dimension.");
    if (!Array.isArray(model.standardization?.mean) || model.standardization.mean.length !== dimension) throw new Error("The embedding model mean does not match its dimension.");
    if (!Array.isArray(model.standardization?.scale) || model.standardization.scale.length !== dimension) throw new Error("The embedding model scale does not match its dimension.");
  }

  function create(model) {
    assertModel(model);
    const dimension = model.embedding_dim;
    const weights = model.weights.map(Number);
    const mean = model.standardization.mean.map(Number);
    const scale = model.standardization.scale.map(Number);
    const baseLogit = Number(model.base_logit);

    function score(embedding) {
      if (!Array.isArray(embedding) || embedding.length !== dimension) throw new Error(`Expected a ${dimension}-dimension embedding.`);
      let logit = baseLogit;
      for (let index = 0; index < dimension; index += 1) {
        const value = Number(embedding[index]);
        if (!Number.isFinite(value)) throw new Error("Embedding values must be finite numbers.");
        logit += ((value - mean[index]) / scale[index]) * weights[index];
      }
      const completionProbability = sigmoid(logit);
      return {
        probability: completionProbability,
        completionProbability,
        terminationProbability: 1 - completionProbability,
        logit,
        completionThreshold: Number(model.classification?.completion_threshold ?? 0.5),
        predictedClass: completionProbability >= Number(model.classification?.completion_threshold ?? 0.5) ? "completed" : "terminated",
        modelType: model.model_type,
        modelVersion: model.version,
        labelDefinition: model.label_definition,
      };
    }

    return { model, score };
  }

  async function sha256Bucket(nctId, bucketCount) {
    if (!global.crypto?.subtle) throw new Error("Secure hashing is unavailable in this browser.");
    const normalizedNctId = String(nctId || "").trim().toUpperCase();
    const digest = new Uint8Array(await global.crypto.subtle.digest("SHA-256", new TextEncoder().encode(normalizedNctId)));
    const leadingValue = (((digest[0] << 24) >>> 0) + (digest[1] << 16) + (digest[2] << 8) + digest[3]);
    return leadingValue % bucketCount;
  }

  async function lookupEmbedding(nctId, lookup) {
    const normalizedNctId = String(nctId || "").trim().toUpperCase();
    if (!normalizedNctId) return null;
    const bucket = await sha256Bucket(normalizedNctId, Number(lookup.bucket_count));
    const path = String(lookup.path_template).replace("{bucket:02d}", String(bucket).padStart(2, "0"));
    const response = await fetch(path, { cache: "force-cache" });
    if (!response.ok) throw new Error("The local NCT embedding lookup could not be loaded.");
    const payload = await response.json();
    const record = (payload.records || []).find((candidate) => candidate.nct_id === normalizedNctId);
    return record?.embedding || null;
  }

  async function load(url = "/static/models/trial-completion-model.json") {
    const response = await fetch(url, { cache: "force-cache" });
    if (!response.ok) throw new Error("The local completion model could not be loaded.");
    return create(await response.json());
  }

  global.TrialEmbeddingScorer = { create, load, lookupEmbedding, sha256Bucket };
})(window);