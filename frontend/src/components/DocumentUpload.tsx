import { useRef, useState } from "react";
import { ApiRequestError, uploadDocument } from "../lib/api";

interface DocumentUploadProps {
  accessToken: string | null;
}

interface IndexedDoc {
  id: string;
  name: string;
  chunks: number;
}

type UploadState =
  | { kind: "idle" }
  | { kind: "uploading"; name: string }
  | { kind: "done"; name: string; chunks: number }
  | { kind: "error"; message: string };

export function DocumentUpload({ accessToken }: DocumentUploadProps) {
  const [state, setState] = useState<UploadState>({ kind: "idle" });
  const [indexed, setIndexed] = useState<IndexedDoc[]>([]);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handleFile(file: File) {
    if (!accessToken) {
      setState({ kind: "error", message: "Sign in before uploading." });
      return;
    }
    setState({ kind: "uploading", name: file.name });
    try {
      const result = await uploadDocument(accessToken, file);
      setIndexed((prev) => [
        ...prev.filter((d) => d.id !== result.document_id),
        { id: result.document_id, name: file.name, chunks: result.chunks },
      ]);
      setState({ kind: "done", name: file.name, chunks: result.chunks });
    } catch (err) {
      const msg = err instanceof ApiRequestError ? err.detail : "Upload failed.";
      setState({ kind: "error", message: msg });
    }
  }

  function onInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) void handleFile(file);
    e.target.value = "";
  }

  function onDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) void handleFile(file);
  }

  const busy = state.kind === "uploading";

  return (
    <div className="panel-scroll">
      <div
        className={`doc-upload__zone${dragging ? " doc-upload__zone--drag" : ""}${busy ? " doc-upload__zone--busy" : ""}`}
        onClick={() => !busy && inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        role="button"
        tabIndex={0}
        aria-label="Upload document"
        onKeyDown={(e) => e.key === "Enter" && !busy && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.docx,.txt"
          style={{ display: "none" }}
          onChange={onInputChange}
        />
        <span className="doc-upload__icon" aria-hidden="true">
          {busy ? "⏳" : "⬆"}
        </span>
        <p className="doc-upload__label">
          {busy
            ? `Indexing ${(state as { name: string }).name}…`
            : "Drop a file or click to upload"}
        </p>
        <p className="doc-upload__hint">PDF · DOCX · TXT</p>
      </div>

      {state.kind === "done" && (
        <p className="doc-upload__feedback doc-upload__feedback--ok">
          {state.name} → {state.chunks} chunks indexed
        </p>
      )}
      {state.kind === "error" && (
        <p className="doc-upload__feedback doc-upload__feedback--err">{state.message}</p>
      )}

      {indexed.length > 0 && (
        <ul className="doc-upload__list">
          {indexed.map((doc) => (
            <li key={doc.id} className="doc-upload__item">
              <span className="doc-upload__item-name">{doc.name}</span>
              <span className="doc-upload__item-meta">{doc.chunks} chunks</span>
            </li>
          ))}
        </ul>
      )}

      {indexed.length === 0 && state.kind === "idle" && (
        <p className="panel-empty">
          Uploaded documents are indexed into the agent's knowledge base and
          available immediately in voice sessions.
        </p>
      )}
    </div>
  );
}
