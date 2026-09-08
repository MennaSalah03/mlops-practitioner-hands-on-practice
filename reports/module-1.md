# Module 1: From notebook to production-ready service

## Metrics of the notebook vs. scripts

* **Notebook:** mae=4.3995 rmse=6.7983
* **Scripts:** mae=4.4617953593057695 rmse=6.784212260444635

This notebook is deliberately messy. It is the "before" picture — commit it as-is and point at it in your report to show what you replaced.

## Serialization Formats

| Format | Human-Readable | Cross-Language | Schema-Enforced | Safe to Load from Untrusted Source |
| --- | --- | --- | --- | --- |
| **JSON** | Yes (plain text) | Yes (universal standard) | Optional (requires external JSON Schema) | Yes (pure data format) |
| **Protobuf** | No (binary) | Yes (compiles to most major languages) | Yes (strict `.proto` contract) | Yes (deserializes static data structures) |
| **Pickle** | No (bytecode/binary) | No (strictly Python ecosystem) | No (arbitrary object serialization) | **No** (executes arbitrary Python code via `__reduce__`) |
| **ONNX** | No (Protobuf-based binary) | Yes (standard runtimes across C++, Python, C#, etc.) | Yes (strictly typed tensor shapes & operator sets) | Yes (declarative computation graph with no arbitrary code execution) |

The Litestar service handles incoming client payloads formatted in **JSON** for effortless HTTP interoperability, while executing inference against a serialized **ONNX** artifact to guarantee high-performance, memory-isolated predictions without the arbitrary code execution vulnerabilities inherent to Python pickles.

*Note: We never loaded the pickle file created for usage in prediction. It was created for initial development and for comparison after having the ONNX serialization.*

## Containerization and Optimization

### Single vs. Multi-Stage Docker Build

| IMAGE | ID | DISK USAGE | CONTENT SIZE | EXTRA |
| --- | --- | --- | --- | --- |
| duration-single_stage | 1f73556f4a56 | 759MB | 177MB |  |
| duration-multi-stage | 53e2aeb8e13b | 742MB | 172MB |  |

While the final image size difference is minimal, the multi-stage build significantly reduces our security attack surface by isolating and discarding system-level build tools, source code, and compilers before creating the final runtime environment.

### Build Context Optimization (`.dockerignore`)

```text
.git
.venv
notebooks/
data/
tests/
__pycache__
*.ipynb

```

By introducing a strict `.dockerignore` file, we prevented the Docker daemon from needlessly uploading local environments and heavy datasets into the build context.

* **Image size without `.dockerignore`:** ~1.8 GB (due to copying `.venv` and `data/`)
* **Image size with `.dockerignore`:** 742 MB

## Maturity Self-Assessment

Based on the standard five-level MLOps maturity model, this repository currently sits at **Level 1 (DevOps but no MLOps)**. While we have containerized the API and scripted the training process, our model training remains a manually triggered event without experiment tracking, version control, or an automated pipeline. To reach Level 2, we need to integrate an experiment tracker, a centralized model registry, and workflow orchestration. *(Spoiler: it is Module 2.)*



```markdown
# Module 1: From notebook to production-ready service

## Metrics of the notebook vs. scripts

* **Notebook:** mae=4.3995 rmse=6.7983 
* **Scripts:** mae=4.4617 rmse=6.7842

This notebook is deliberately messy. It is the "before" picture — commit it as-is and point at it in your report to show what you replaced.

## Serialization Architecture

| Format | Human-Readable | Cross-Language | Schema-Enforced | Safe to Load from Untrusted Source |
| :--- | :--- | :--- | :--- | :--- |
| **JSON** | Yes (plain text) | Yes (universal standard) | Optional (requires external JSON Schema) | Yes (pure data format) |
| **Protobuf** | No (binary) | Yes (compiles to most major languages) | Yes (strict `.proto` contract) | Yes (deserializes static data structures) |
| **Pickle** | No (bytecode/binary) | No (strictly Python ecosystem) | No (arbitrary object serialization) | **No** (executes arbitrary Python code via `__reduce__`) |
| **ONNX** | No (Protobuf-based binary) | Yes (standard runtimes across C++, Python, C#, etc.) | Yes (strictly typed tensor shapes & operator sets) | Yes (declarative computation graph with no arbitrary code execution) |

The Litestar service handles incoming client payloads formatted in **JSON** for effortless HTTP interoperability. Inference is executed against a serialized **ONNX** artifact to guarantee high-performance, memory-isolated predictions without the arbitrary code execution vulnerabilities inherent to Python pickles. 

*Note: The complete pickle artifact (including the model) was generated for initial development and for strict mathematical parity testing ($10^{-4}$ tolerance) against the ONNX graph, but it is not loaded by the prediction endpoint.*

## Containerization and Optimization

### Build Architecture Comparison

| Image | ID | Disk Usage | Content Size | Extra |
| :--- | :--- | :--- | :--- | :--- |
| duration-single_stage | 1f73556f4a56 | 759MB | 177MB | |
| duration-multi-stage | 53e2aeb8e13b | 742MB | 172MB | |

While the final image size difference is minimal, the multi-stage build significantly reduces the attack surface by isolating and discarding system-level build tools, standard Python packaging utilities, and source code compilers before creating the final runtime environment.

### Build Context Optimization (`.dockerignore`)

```text
.git
.venv
notebooks/
data/
tests/
__pycache__
*.ipynb

```

Implementing a strict `.dockerignore` file prevented the Docker daemon from needlessly pulling local environments and heavy datasets into the build context.


## Maturity Self-Assessment

Based on the standard five-level MLOps maturity model, this repository currently sits at **Level 1 (DevOps but no MLOps)**. We have containerized the API, scripted the training process, and implemented isolated dependency management, but model training remains a manually triggered, untracked event. To reach Level 2, we must integrate automated experiment tracking, a centralized model registry, and a reproducible data pipeline.

```
