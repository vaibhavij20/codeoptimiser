export default function ReportPanel({ report }) {
  if (!report) return null;

  return (
    <div style={{
      marginTop: 20,
      border: "1px solid #2e2e2e",
      borderRadius: 8,
      overflow: "hidden",
    }}>
      <div style={{
        padding: "10px 14px",
        fontSize: 13,
        fontWeight: 500,
        background: "#1a1a1a",
        color: "#e5e5e5",
        borderBottom: "1px solid #2e2e2e",
        display: "flex",
        alignItems: "center",
        gap: 8,
      }}>
        📋 Optimization report
      </div>
      <pre style={{
        margin: 0,
        padding: "14px 16px",
        fontFamily: "sans-serif",
        fontSize: 13,
        lineHeight: 1.7,
        whiteSpace: "pre-wrap",
        wordBreak: "break-word",
        background: "#1e1e1e",
        color: "#d4d4d4",
      }}>
        {report}
      </pre>
    </div>
  );
}