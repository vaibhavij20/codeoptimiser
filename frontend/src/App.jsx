import { useState } from "react";
import CodeEditor from "./components/CodeEditor";
import OptimizeButton from "./components/OptimizeButton";
import DiffPanel from "./components/DiffPanel";
import ReportPanel from "./components/ReportPanel";
import { optimizeCode } from "./api";

const DEFAULT_CODE = `result=[]
for x in range(100):
    result.append(x*x)
print(result)`;

export default function App() {
  const [code, setCode] = useState(DEFAULT_CODE);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  async function handleOptimize() {
    try {
      setLoading(true);
      setResult(null);
      const response = await optimizeCode(code);
      setResult(response);
    } catch (error) {
      console.error(error);
      alert("Optimization failed. Check the console for details.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ padding: "24px 32px", minHeight: "100vh", background: "#141414" }}>
      <h1 style={{ fontSize: 22, fontWeight: 500, marginBottom: 20, color: "#e5e5e5" }}>
         Code Optimizer
      </h1>

      <CodeEditor code={code} setCode={setCode} />

      <OptimizeButton loading={loading} onClick={handleOptimize} />

      {result?.analysis && (
        <div style={{
          marginTop: 20,
          padding: "14px 16px",
          background: "#1a1a1a",
          border: "1px solid #2e2e2e",
          borderRadius: 8,
        }}>
          <h2 style={{ fontSize: 15, fontWeight: 500, marginBottom: 8, color: "#e5e5e5" }}>
            Analysis
          </h2>
          <p style={{ fontSize: 13, lineHeight: 1.7, color: "#9ca3af", margin: 0 }}>
            {result.analysis}
          </p>
        </div>
      )}

      {result?.optimized_code && (
        <DiffPanel oldCode={code} newCode={result.optimized_code} />
      )}

      <ReportPanel report={result?.final_report} />
    </div>
  );
}