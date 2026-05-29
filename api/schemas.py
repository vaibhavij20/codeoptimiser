from pydantic import BaseModel


class OptimizeRequest(BaseModel):
    code: str
    language: str = "python"


class OptimizeResponse(BaseModel):
    analysis: str
    optimized_code: str | None = None
    final_report: str