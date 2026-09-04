### Steps

```
pip install -e ".[dev]" # install
ruff check src tests && black --check src tests # lint
pytest -v --cov=src/prodml --cov-report=term-missing # test
python -m prodml.train # train
uvicorn prodml.api.main:app --reload --port 8000 # serve
```
Note: development was done on an ubuntu 24.0 machine
