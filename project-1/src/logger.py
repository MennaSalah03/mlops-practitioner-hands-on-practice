import os
import structlog
import logging

def configure_logging():
    """Sets the global rules for how structlog behaves."""
    # Check an environment variable to see if we are in Docker or Local
    env = os.getenv("ENVIRONMENT", "local")

    # Common processors (things we want in EVERY log)
    shared_processors = [
        structlog.processors.TimeStamper(fmt="iso"), # Adds a timestamp
        structlog.stdlib.add_log_level,              # Adds INFO, DEBUG, etc.
    ]

    # Environment-specific formatting
    if env == "production":
        # Docker/Cloud mode: Output raw JSON
        formatter = structlog.processors.JSONRenderer()
    else:
        # Local mode: Output pretty, color-coded text for the human reading the terminal
        formatter = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=shared_processors + [formatter],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
