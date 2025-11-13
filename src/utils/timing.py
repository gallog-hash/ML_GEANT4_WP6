"""
Timing utilities for optional execution time measurement.

This module provides a context manager for measuring execution time
across different pipeline components. The timing functionality is
optional and can be enabled/disabled via configuration.
"""


class OptionalTimer:
    """
    Context manager for optional execution timing.

    This timer can be used to measure execution time of code blocks
    across different pipeline components. It gracefully handles cases
    where timing is disabled or unavailable.

    Args:
        enabled (bool): Whether timing is enabled. Default: True
        logger: Logger instance for output. If None, prints to stdout.
        description (str): Description of the timed operation.
            Default: "Execution"

    Example:
        >>> with OptionalTimer(enabled=True, logger=logger,
        ...                     description="Data loading"):
        ...     data = load_data()
        ...
        # Output: "Data loading completed in 2.34s"
    """

    def __init__(self, enabled=True, logger=None, description="Execution"):
        self.enabled = enabled
        # Handle both VAELogger wrapper and actual logging.Logger
        if logger is not None and hasattr(logger, "get_logger"):
            self.logger = logger.get_logger()
        else:
            self.logger = logger
        self.description = description
        self.start_time = None

    def __enter__(self):
        """Start timing when entering context."""
        if not self.enabled:
            return self

        try:
            import time

            self.start_time = time.perf_counter()
        except ImportError:
            if self.logger:
                self.logger.warning("Timing unavailable: time module not found")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log elapsed time when exiting context."""
        if not self.enabled or self.start_time is None:
            return

        try:
            import time

            elapsed = time.perf_counter() - self.start_time

            # Format time with appropriate precision based on duration
            if elapsed >= 1.0:
                # For times >= 1s, show seconds with millisecond precision
                msg = f"{self.description} completed in {elapsed:.3f}s"
            elif elapsed >= 0.001:
                # For times >= 1ms, show milliseconds with microsecond
                # precision
                elapsed_ms = elapsed * 1000
                msg = f"{self.description} completed in {elapsed_ms:.3f}ms"
            else:
                # For times < 1ms, show microseconds with nanosecond
                # precision
                elapsed_us = elapsed * 1_000_000
                msg = f"{self.description} completed in {elapsed_us:.3f}µs"

            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)
        except Exception:
            # Silently fail if timing can't complete
            pass
