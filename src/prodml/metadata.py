"""generate the metdata for model"""

import hashlib
import pickle
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

import sklearn

from prodml.config import config


@dataclass
class ArtifactMetadata:
    """Everything you'd want to know about an artifact without deserialization."""

    model_version: str
    training_date: str
    feature_names: list[str]
    framework: str
    metrics: dict
    artifact_hash: str = field(default="")

    def to_dict(self) -> dict:
        """return instance as dictionary"""
        return asdict(self)


def _git_sha(short: bool = True) -> str:
    """Best-effort git commit sha, used as the model version. Falls back to 'unknown'."""
    try:
        args = (
            ["git", "rev-parse", "--short", "HEAD"]
            if short
            else ["git", "rev-parse", "HEAD"]
        )
        return subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _hash_payload(payload: dict) -> str:
    """Deterministic hash of the model payload (dv + model), used as artifact_hash.

    Hashing the pickled bytes of just the payload (not the metadata) keeps the
    hash stable across re-persists and avoids the "hash of the hash" problem.
    """
    return hashlib.sha256(
        pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()


def build_metadata(payload: dict, metrics: dict) -> ArtifactMetadata:
    """building the metadata as dictionary"""
    return ArtifactMetadata(
        model_version=_git_sha(),
        training_date=datetime.now(UTC).isoformat(),
        feature_names=config.raw_input_fields,
        framework=f"sklearn-{sklearn.__version__}",
        metrics=metrics,
        artifact_hash=_hash_payload(payload),
    )
