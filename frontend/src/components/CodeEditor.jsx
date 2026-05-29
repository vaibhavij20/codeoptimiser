import Editor from "@monaco-editor/react";

export default function CodeEditor({ code, setCode }) {
  return (
    <div style={{
      border: "1px solid #2e2e2e",
      borderRadius: 8,
      overflow: "hidden",
      width: "100%",
    }}>
      <Editor
        height="300px"
        width="100%"
        theme="vs-dark"
        defaultLanguage="python"
        value={code}
        onChange={(value) => setCode(value || "")}
        options={{
          fontSize: 13,
          lineHeight: 1.6,
          minimap: { enabled: false },
          scrollBeyondLastLine: false,
          wordWrap: "on",
          padding: { top: 12, bottom: 12 },
          renderLineHighlight: "none",
          overviewRulerLanes: 0,
          hideCursorInOverviewRuler: true,
          scrollbar: {
            vertical: "auto",
            horizontal: "hidden",
          },
        }}
      />
    </div>
  );
}