/**
 * Tests for detection method catalog (GET /api/jobs/presets) and
 * modelType wire/Firestore contracts on POST /api/jobs/start-job.
 */

import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import express from 'express';
import request from 'supertest';

const mockFirestoreCreate = jest.fn<(...args: any[]) => Promise<void>>();
const mockPublishMessage = jest.fn<(...args: any[]) => Promise<string>>();

jest.unstable_mockModule('../../config/gcp-clients', () => ({
  storage: {
    bucket: jest.fn(() => ({
      file: jest.fn(() => ({
        getSignedUrl: jest.fn(),
        delete: jest.fn(),
      })),
    })),
  },
  firestore: {
    collection: jest.fn(() => ({
      doc: jest.fn(() => ({
        create: mockFirestoreCreate,
        get: jest.fn(),
        update: jest.fn(),
      })),
    })),
  },
  pubsub: {
    topic: jest.fn(() => ({
      publishMessage: mockPublishMessage,
    })),
  },
}));

jest.unstable_mockModule('../../middleware/auth', () => ({
  requireAuth: (req: any, _res: any, next: any) => {
    req.user = { id: 'user-123', email: 'researcher@nyu.edu' };
    next();
  },
}));

jest.unstable_mockModule('../../middleware/rateLimits', () => ({
  uploadLimiter: (_req: any, _res: any, next: any) => next(),
  uploadUrlLimiter: (_req: any, _res: any, next: any) => next(),
  pollLimiter: (_req: any, _res: any, next: any) => next(),
  downloadLimiter: (_req: any, _res: any, next: any) => next(),
}));

jest.unstable_mockModule('../../services/vertexAi', () => ({
  cancelVertexAIJob: jest.fn<(...args: any[]) => Promise<boolean>>().mockResolvedValue(true),
}));

jest.unstable_mockModule('../../config/logger', () => ({
  logger: {
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
  },
}));

describe('Jobs presets & start-job modelType contract', () => {
  const validJobId = '123e4567-e89b-12d3-a456-426614174000';
  const validGcsFile = 'uploads/user-123/survey.csv';

  beforeEach(() => {
    jest.clearAllMocks();
    mockFirestoreCreate.mockResolvedValue(undefined);
    mockPublishMessage.mockResolvedValue('msg-id-1');
  });

  async function buildApp() {
    const { jobsRouter } = await import('../../routes/jobs');
    const app = express();
    app.use(express.json());
    app.use('/api/jobs', jobsRouter);
    return app;
  }

  it('GET /api/jobs/presets returns both presets and detection methods', async () => {
    const app = await buildApp();
    const res = await request(app).get('/api/jobs/presets');

    expect(res.status).toBe(200);
    expect(res.body.presets.map((p: any) => p.id)).toEqual([
      'auto',
      'small',
      'medium',
      'large',
    ]);
    expect(res.body.methods.map((m: any) => m.id)).toEqual([
      'chow_liu',
      'autoencoder',
    ]);
  });

  it('POST /api/jobs/start-job with chow_liu persists modelType and suppresses modelPreset', async () => {
    const app = await buildApp();
    const res = await request(app).post('/api/jobs/start-job').send({
      jobId: validJobId,
      gcsFileName: validGcsFile,
      modelType: 'chow_liu',
      modelPreset: 'large',
    });

    expect(res.status).toBe(200);
    expect(mockFirestoreCreate).toHaveBeenCalledTimes(1);
    const createdDoc = mockFirestoreCreate.mock.calls[0][0];
    expect(createdDoc.modelType).toBe('chow_liu');
    expect(createdDoc.modelPresetRequested).toBeUndefined();

    expect(mockPublishMessage).toHaveBeenCalledTimes(1);
    const publishedBuf = mockPublishMessage.mock.calls[0][0].data as Buffer;
    const published = JSON.parse(publishedBuf.toString('utf-8'));
    expect(published.modelType).toBe('chow_liu');
    expect(published.modelPreset).toBeUndefined();
  });

  it('POST /api/jobs/start-job with autoencoder omits default modelType from Pub/Sub for rolling deploys', async () => {
    const app = await buildApp();
    const res = await request(app).post('/api/jobs/start-job').send({
      jobId: validJobId,
      gcsFileName: validGcsFile,
      modelType: 'autoencoder',
      modelPreset: 'medium',
    });

    expect(res.status).toBe(200);
    const createdDoc = mockFirestoreCreate.mock.calls[0][0];
    expect(createdDoc.modelType).toBe('autoencoder');
    expect(createdDoc.modelPresetRequested).toBe('medium');

    const publishedBuf = mockPublishMessage.mock.calls[0][0].data as Buffer;
    const published = JSON.parse(publishedBuf.toString('utf-8'));
    expect(published.modelType).toBeUndefined();
    expect(published.modelPreset).toBe('medium');
  });

  it('POST /api/jobs/start-job omitting modelType preserves legacy defaults', async () => {
    const app = await buildApp();
    const res = await request(app).post('/api/jobs/start-job').send({
      jobId: validJobId,
      gcsFileName: validGcsFile,
    });

    expect(res.status).toBe(200);
    const createdDoc = mockFirestoreCreate.mock.calls[0][0];
    expect(createdDoc.modelType).toBe('autoencoder');
    expect(createdDoc.modelPresetRequested).toBe('auto');

    const publishedBuf = mockPublishMessage.mock.calls[0][0].data as Buffer;
    const published = JSON.parse(publishedBuf.toString('utf-8'));
    expect(published.modelType).toBeUndefined();
    expect(published.modelPreset).toBeUndefined();
  });
});
