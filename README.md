# NYC Taxi Trip Duration Predictor

This repository is an attempt to have a production-ready machine learning service that predicts NYC taxi trip durations. It transitions a raw EDA notebook into a robust, containerized MLOps architecture. The pipeline features scikit-learn for feature engineering, ONNX for high-performance and secure model inference, and Litestar for a highly concurrent API, all neatly packaged into a multi-stage Docker container.

## Quick Start

### Option 1: Run the Pre-built API
If you just want to test the inference endpoint, pull the lightweight runtime image directly from DockerHub:

```
docker run --rm -p 8000:8000 menna011/prodml-api:0.1.0
```

### Option 2: Build and Develop Locally

To train the model, export the ONNX artifacts, and compile the container from scratch:

```
# 1. Install dependencies (including dev tools)
pip install -e ".[dev]"

# 2. Train the model and serialize to ONNX/Pickle
python -m prodml.export

# 3. Run the Litestar API
uvicorn prodml.api.main:app --reload --port 8000

```

## API Usage & Documentation

Once the container is running, the Litestar API is available at `http://localhost:8000`.

**Interactive API Docs:** Navigate to `http://localhost:8000/schema/swagger` in your browser to view the auto-generated Swagger UI and test the endpoints visually.



**The API includes 4 Endpoints:**
- **GET:**
  - `/health`: gets the health status of knobs (at a later stage, but currently is hardcoded as "healthy")
  - `/metadata`: shows the loaded model's metadata.
- **POST:**
  - `/predict`: takes feature dictionary and outputs the prediction.
  - `/predict/batch`: takes a list of dictionaries with features and outputs a list of predictions


### Example Request

Test the `/predict` endpoint using `curl`:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "PULocationID": 43,
        "DOLocationID": 236,
        "trip_distance": 3.5
      }'

```




## Testing

This project includes a test suite at `tests/`. To run them locally with coverage reporting:

```bash
pytest -v --cov=src/prodml --cov-report=term-missing

```

---

## Repository Structure

```text
.
├── data/                  # Raw and processed datasets
├── docker/
│   ├── docker-compose.yml # Container orchestration
│   └── Dockerfile         # Multi-stage build instructions
├── models/                # Serialized ONNX and Pickle artifacts
├── notebooks/             # Exploratory data analysis
├── reports/               # Module assignments and progress reports
├── src/
│   └── prodml/
│       ├── api/           # Litestar API routes and schemas
│       ├── config.py      # Environment variables and configuration
│       ├── data.py        # Data loading and splitting
│       ├── features.py    # DictVectorizer and feature engineering
│       ├── predict.py     # ONNX inference runner
│       └── train.py       # Model training and artifact export
├── tests/                 # Pytest suite (parity tests, API tests)
├── .dockerignore          # Docker build exclusions
├── .gitignore             # Git exclusions
├── pyproject.toml         # Python dependencies and metadata
└── README.md              # Project documentation
```
