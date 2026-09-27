import type {
  DevTokenRequest,
  DevTokenResponse,
  LiveKitTokenRequest,
  LiveKitTokenResponse,
} from "../types/auth";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export class ApiRequestError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiRequestError";
    this.status = status;
    this.detail = detail;
  }
}

async function parseErrorDetail(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: string };
    if (body && typeof body.detail === "string") return body.detail;
  } catch {
    // response wasn't JSON; fall through to the status text below
  }
  return response.statusText || `Request failed with status ${response.status}`;
}

/**
 * Dev-only bootstrap: POST /v1/dev/token. Only reachable when the backend is
 * running with ENVIRONMENT=development (see token_service.py) -- a 404 here
 * means the backend has that endpoint disabled, not that the URL is wrong.
 */
export async function fetchDevToken(body: DevTokenRequest): Promise<DevTokenResponse> {
  const response = await fetch(`${API_BASE_URL}/v1/dev/token`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new ApiRequestError(response.status, await parseErrorDetail(response));
  }

  return (await response.json()) as DevTokenResponse;
}

/**
 * Production token exchange: POST /v1/livekit/token with the application JWT
 * as a bearer token. Used to mint a fresh LiveKit token for a new room
 * without re-running the dev bootstrap.
 */
export async function fetchLiveKitToken(
  accessToken: string,
  body: LiveKitTokenRequest,
): Promise<LiveKitTokenResponse> {
  const response = await fetch(`${API_BASE_URL}/v1/livekit/token`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${accessToken}`,
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new ApiRequestError(response.status, await parseErrorDetail(response));
  }

  return (await response.json()) as LiveKitTokenResponse;
}

export const LIVEKIT_URL = import.meta.env.VITE_LIVEKIT_URL ?? "";

export interface UploadDocumentResponse {
  document_id: string;
  chunks: number;
  status: string;
}

export async function uploadDocument(
  accessToken: string,
  file: File,
): Promise<UploadDocumentResponse> {
  const form = new FormData();
  form.append("file", file);

  const response = await fetch(`${API_BASE_URL}/v1/documents/upload`, {
    method: "POST",
    // No Content-Type header — browser sets multipart/form-data + boundary automatically.
    headers: { Authorization: `Bearer ${accessToken}` },
    body: form,
  });

  if (!response.ok) {
    throw new ApiRequestError(response.status, await parseErrorDetail(response));
  }

  return (await response.json()) as UploadDocumentResponse;
}

export const API_URL = API_BASE_URL;

async function postJson<T>(path: string, accessToken: string, body?: unknown): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${accessToken}` },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    throw new ApiRequestError(response.status, await parseErrorDetail(response));
  }
  return (await response.json()) as T;
}

export interface AgentKeyResponse {
  agent_key: string;
  expires_in: number;
  tenant_id: string;
}

export function createAgentKey(accessToken: string): Promise<AgentKeyResponse> {
  return postJson<AgentKeyResponse>("/v1/agent-keys", accessToken);
}

export interface KnowledgeHit {
  text: string;
  score: number | null;
  source: string | null;
  chunk: string | null;
}

export interface KnowledgeQueryResponse {
  query: string;
  hits: KnowledgeHit[];
  took_ms: number;
  index_ready: boolean;
}

export function queryKnowledge(accessToken: string, query: string, topK = 3): Promise<KnowledgeQueryResponse> {
  return postJson<KnowledgeQueryResponse>("/v1/knowledge/query", accessToken, { query, top_k: topK });
}
