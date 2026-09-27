import { useState } from "react";

interface ChatInputProps {
  isConnected: boolean;
  onSend: (text: string) => Promise<void>;
}

export function ChatInput({ isConnected, onSend }: ChatInputProps) {
  const [text, setText] = useState("");
  const [sending, setSending] = useState(false);
  const canSend = isConnected && !sending && text.trim().length > 0;

  async function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!canSend) return;
    setSending(true);
    try {
      await onSend(text);
      setText("");
    } finally {
      setSending(false);
    }
  }

  return (
    <form className="chat-input" onSubmit={submit}>
      <input
        className="field__input chat-input__field"
        type="text"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder={isConnected ? "Type a message…" : "Connect to start chatting"}
        disabled={!isConnected}
        aria-label="Message the agent"
        maxLength={500}
      />
      <button className="btn btn--primary" type="submit" disabled={!canSend}>
        Send
      </button>
    </form>
  );
}
