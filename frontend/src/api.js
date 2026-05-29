import axios from "axios";

const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:5001";

export async function optimizeCode(code) {
  const response = await axios.post(
    `${API_URL}/optimize`,
    { code }
  );

  return response.data;
}