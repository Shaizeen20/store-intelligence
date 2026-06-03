# 💜 Store Intelligence: Agentic AI Telemetry & Real-Time Conversion Pipeline

An enterprise-grade, real-time analytics engine and live telemetry command surface engineered for high-density physical retail environments. By dynamically correlating high-throughput computer vision spatial tracking logs with transactional Point-of-Sale (POS) event frameworks, the platform completely eliminates traditional brick-and-mortar analytical blind spots and models true offline conversion rates.

---

## 🛠️ Key Architectural Triumphs

- **Lock-Free Analytical Pipeline:** Engineered an asynchronous, SQLite-backed ingestion engine utilizing a sliding-window data architecture to continuously compute metrics under intensive concurrent traffic loads.
- **Write-Ahead Logging (WAL) & Lock Isolation:** Configured native SQLite WAL execution patterns mapped to a module-level `asyncio.Lock()` structure (`_ingest_commit_lock`) to serialize transactional commits, completely eliminating race conditions.
- **🔮 Gemini LLM-as-a-Judge Anomaly Tier:** Integrated an advanced, asynchronous `google-genai` SDK worker node that continually evaluates tracking data against statistical Z-scores to catch, diagnose, and audit camera frame-drops or funnel drops.
- **🤖 Autonomous Agentic Mitigations:** Built automated mitigation workflows that simulate real-time corporate operational fixes (e.g., auto-dispatching node cache flushes or alerting supervisors to open billing queues) paired with a live, fractional-cent per audit execution cost tracker.
- **Elite Test Engineering Rigor:** Achieved a rock-solid, production-grade **87% total statement coverage baseline** across the entire core system application and processing packages, verified by a 60-test automation suite.

---

## 🚀 Technical Core Stack

| Layer | Technology |
|---|---|
| **Backend Application** | FastAPI, Asyncio, SQLAlchemy, Uvicorn, Pydantic v2 |
| **Frontend Command Surface** | Streamlit, Plotly Engine (High-Contrast Native Visualizations) |
| **Verification Matrix** | Pytest, Pytest-Asyncio, Pytest-Cov |
| **Deployment Vector** | Docker, Multi-Stage Build Layered Docker Compose |

---

## 🧪 Automated Verification Test Suite

The system maintains a comprehensive test suite covering ingestion edge cases, data sanitisation, duplication cache rollbacks, metrics pooling, and anomaly thresholds.

To run the verification checks natively, execute:

```bash
# Install core and testing packages
python -m pip install -r requirements.txt

# Run the complete test execution path with coverage logging
python -m pytest tests/ --cov=app --cov=pipeline
```

**Verified Test Run Metrics:**

```
tests/test_anomalies.py ..............                                   [ 23%]
tests/test_ingestion.py ...........                                      [ 41%]
tests/test_metrics.py  ..................                                 [ 71%]
tests/test_pipeline.py .................                                 [100%]

Name                     Stmts   Miss  Cover
--------------------------------------------
TOTAL                     1066    138    87%
======================== 60 passed in 4.53s ========================
```

---

## 🐳 Production Deployment via Docker Compose

The entire multi-tier ecosystem is fully containerised and deployable via a single orchestration layer.

### 1. Initialize the Services

From the repository root directory, open your terminal and spin up the isolated container layout:

```bash
docker compose up --build
```

Docker will automatically install the system dependencies, compile the optimised Python caching mirrors, run internal health checks on the FastAPI core layer, and securely launch the interactive user cockpit.

### 2. Network Endpoint Mapping

| Surface | URL |
|---|---|
| **Premium Executive Dashboard UI** | http://localhost:8501 |
| **Core Analytics & Ingestion Docs** | http://localhost:8000/docs |
| **Automated Node Health Probe** | http://localhost:8000/health |

---

## ⚙️ Activating Live AI Inferences (Optional)

By default, the presentation cockpit mounts deterministic operational fallback data frames (`_DEMO_METRICS`, `_DEMO_FUNNEL`) to guarantee lightning-fast response times and prevent lag during demo screen recordings.

To shift the entire platform into live production mode where background workers query **gemini-2.5-flash** dynamically using live context maps, set your credential keys in your environment before launching the containers:

```powershell
# Windows PowerShell
$env:GEMINI_API_KEY="AIzaSyYourActualProductionGeminiAPIKeyHere"
```

```bash
# Linux / macOS
export GEMINI_API_KEY="AIzaSyYourActualProductionGeminiAPIKeyHere"
```

Once injected, the conversational sidebar chat box and the anomaly engines switch over completely to real-time, live model inferences.
