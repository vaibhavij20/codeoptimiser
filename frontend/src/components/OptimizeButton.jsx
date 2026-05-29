export default function OptimizeButton({ onClick, loading }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 16 }}>
      <button
        onClick={onClick}
        disabled={loading}
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 8,
          padding: "9px 20px",
          fontSize: 14,
          fontWeight: 500,
          background: "#1e1e1e",
          color: "#e5e5e5",
          border: "1px solid #3e3e3e",
          borderRadius: 8,
          cursor: loading ? "not-allowed" : "pointer",
          opacity: loading ? 0.5 : 1,
          transition: "background 0.15s",
        }}
      >
         {loading ? "Optimizing…" : "Optimize code"}
      </button>

      {loading && (
        <span style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 6,
          fontSize: 12,
          color: "#9ca3af",
        }}>
          <span style={{
            width: 12,
            height: 12,
            border: "2px solid #3e3e3e",
            borderTopColor: "#9ca3af",
            borderRadius: "50%",
            display: "inline-block",
            animation: "spin 0.7s linear infinite",
          }} />
          Working…
        </span>
      )}

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}