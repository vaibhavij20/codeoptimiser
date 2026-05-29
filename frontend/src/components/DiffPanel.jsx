export default function DiffPanel({ oldCode, newCode }) {
  const pane = (label, code, accentColor) => (
    <div style={{
      border: "1px solid #2e2e2e",
      borderRadius: 8,
      overflow: "hidden",
    }}>
      <div style={{
        padding: "8px 14px",
        fontSize: 12,
        fontWeight: 500,
        letterSpacing: "0.03em",
        textTransform: "uppercase",
        background: "#1a1a1a",
        color: accentColor,
        borderBottom: "1px solid #2e2e2e",
      }}>
        {label}
      </div>
      <pre style={{
        margin: 0,
        padding: "14px 16px",
        fontFamily: "monospace",
        fontSize: 12,
        lineHeight: 1.6,
        whiteSpace: "pre-wrap",
        wordBreak: "break-word",
        background: "#1e1e1e",
        color: "#d4d4d4",
        maxHeight: 260,
        overflowY: "auto",
      }}>
        {code}
      </pre>
    </div>
  );

  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: "minmax(0,1fr) minmax(0,1fr)",
      gap: 16,
      marginTop: 20,
    }}>
      {pane("Original", oldCode, "#9ca3af")}
      {pane("Optimized", newCode, "#4ade80")}
    </div>
  );
}