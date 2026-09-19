# 01 · Serving Models (FastAPI + Docker)

> **Goal:** Turn a notebook model into a reliable API. This is what makes you an *engineer*, not just a modeler.

## Project Structure
```
ml-service/
├── app/
│   ├── main.py          # FastAPI app
│   ├── schemas.py       # Pydantic request/response models
│   └── model.py         # load model once, predict
├── models/model.joblib
├── tests/test_api.py
├── requirements.txt
└── Dockerfile
```

## FastAPI Service
```python
# app/main.py
from contextlib import asynccontextmanager
import joblib, pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

class Passenger(BaseModel):
    age: float | None = Field(None, ge=0, le=110)
    fare: float = Field(..., ge=0)
    sibsp: int = 0
    parch: int = 0
    sex: str
    pclass: str = Field(..., alias="class")
    embarked: str | None = None
    who: str

class Prediction(BaseModel):
    survived: bool
    probability: float

ml = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml["model"] = joblib.load("models/model.joblib")   # load ONCE at startup
    yield
    ml.clear()

app = FastAPI(title="Titanic Model API", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=Prediction)
def predict(p: Passenger):
    X = pd.DataFrame([p.model_dump(by_alias=True)])
    prob = float(ml["model"].predict_proba(X)[0, 1])
    return Prediction(survived=prob >= 0.5, probability=round(prob, 4))
```
Run: `uvicorn app.main:app --reload` → open `http://localhost:8000/docs`.

## Docker
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ app/
COPY models/ models/
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
```bash
docker build -t titanic-api . && docker run -p 8000:8000 titanic-api
```

## Test It
```python
# tests/test_api.py
from fastapi.testclient import TestClient
from app.main import app

def test_predict():
    with TestClient(app) as client:   # triggers lifespan → model loads
        r = client.post("/predict", json={"age": 22, "fare": 7.25, "sex": "male",
                                          "class": "Third", "who": "man", "embarked": "S"})
        assert r.status_code == 200
        assert 0 <= r.json()["probability"] <= 1
```

## Serving Options
| Need | Tool |
|---|---|
| Simple REST | FastAPI (this file) |
| Batch predictions | Scheduled job (Airflow, cron) writing to DB |
| High-throughput DL | ONNX Runtime, TorchServe, NVIDIA Triton |
| LLM serving | vLLM, TGI, Ollama |
| Managed | AWS SageMaker, GCP Vertex AI, Azure ML, Hugging Face Endpoints |

**Tip:** Java/Spring Boot backends can call this Python service over HTTP — a common real-world split.

## Exercises
1. Serve your best Stage-02 model with FastAPI + Docker; add input validation and tests.
2. Add a `/predict/batch` endpoint that accepts a list and returns a list.

---
Next → [MLOps Lifecycle](02-mlops-lifecycle.md)
