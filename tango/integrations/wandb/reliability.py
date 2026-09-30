"""Bounded retries for W&B transport failures, including SDK-wrapped errors."""

import logging
import math
import os
import random
import re
import time
from functools import wraps
from typing import Any, Callable, Dict, Iterator, Optional, TypeVar

import requests
import wandb
from wandb.sdk.artifacts.exceptions import WaitTimeoutError

T = TypeVar("T")
logger = logging.getLogger(__name__)
_retry_random = random.SystemRandom()


def error_chain(error: BaseException) -> Iterator[BaseException]:
    """Include transport errors hidden by W&B's AuthenticationError / CommError."""
    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for nested in (current.__cause__, current.__context__, getattr(current, "exc", None)):
            if isinstance(nested, BaseException):
                pending.append(nested)


def http_status(error: BaseException) -> Optional[int]:
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def is_retryable(error: BaseException, *, run_id: Optional[str] = None) -> bool:
    errors = list(error_chain(error))
    statuses = [http_status(item) for item in errors]
    # A known permanent HTTP response takes precedence over a wrapper's type.
    if any(status in {400, 401, 403, 404, 409, 422} for status in statuses):
        if 404 not in statuses or any(status not in {None, 404} for status in statuses):
            return False
        if not run_id:
            return False
        for item in errors:
            response = getattr(item, "response", None)
            message = (str(item) + " " + str(getattr(response, "text", ""))).lower()
            if re.search(
                r"(?<![\w-])" + re.escape(run_id.lower()) + r"(?![\w-])", message
            ) and re.search(r"(?<![\w-])(?:failed to find run|run not found)\b", message):
                return True
        return False
    if any(
        status == 408 or status == 429 or (status is not None and 500 <= status < 600)
        for status in statuses
    ):
        return True
    if any(status is not None and status >= 400 for status in statuses):
        return False
    if any(
        isinstance(
            item, (requests.ConnectionError, requests.Timeout, ConnectionError, TimeoutError)
        )
        for item in errors
    ):
        return True
    for item in errors:
        # W&B 0.22.2 suppresses the ConnectionError cause in this one case.
        if isinstance(item, wandb.errors.AuthenticationError):
            return str(item) == "Unable to connect to server to verify API token."
    # Do not mistake missing artifacts or rejected credentials for communication failures.
    message = " ".join(str(item).lower() for item in errors)
    if any(
        text in message
        for text in (
            "not found",
            "does not contain artifact",
            "unable to fetch artifact with name",
            "permission denied",
            "unauthorized",
            "forbidden",
            "invalid api key",
        )
    ):
        return False
    return any(isinstance(item, wandb.errors.CommError) for item in errors)


def wandb_retry(
    operation: Callable[[], T],
    *,
    description: str,
    run_id: Optional[str] = None,
    max_attempts: int = 6,
    max_elapsed: float = 120.0,
    retry_if: Optional[Callable[[Exception], bool]] = None,
) -> T:
    """Retry the operation, never a whole step; SDK calls retain their own timeouts."""
    if max_attempts < 1 or max_elapsed <= 0:
        raise ValueError("Retry attempts and elapsed-time budget must be positive")
    started = time.monotonic()
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as error:
            if (
                attempt == max_attempts
                or not is_retryable(error, run_id=run_id)
                or (retry_if is not None and not retry_if(error))
            ):
                raise
            delay = min(2**attempt, 30) + _retry_random.uniform(0, 1)
            for nested in error_chain(error):
                response = getattr(nested, "response", None)
                retry_after = getattr(response, "headers", {}).get("Retry-After")
                if retry_after is not None:
                    try:
                        delay = max(delay, float(retry_after))
                    except (TypeError, ValueError):
                        pass
            if time.monotonic() - started + delay >= max_elapsed:
                raise
            logger.warning(
                "W&B %s failed (%s), attempt %d/%d; retrying in %.1fs",
                description,
                type(error).__name__,
                attempt,
                max_attempts,
                delay,
            )
            time.sleep(delay)
    raise AssertionError("Retry loop exited without returning or raising")


def wait_for_artifact(
    artifact: wandb.Artifact, *, description: str, timeout: Optional[int] = None
) -> None:
    """Wait for completion without counting pending uploads as transport failures."""
    deadline = None if timeout is None else time.monotonic() + timeout

    def remaining_time() -> Optional[float]:
        if deadline is None:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WaitTimeoutError(f"W&B {description} did not complete within {timeout} seconds")
        return remaining

    def poll() -> bool:
        remaining = remaining_time()
        # The SDK takes whole seconds; round up to avoid a zero-second busy loop.
        poll_timeout = 30 if remaining is None else min(30, math.ceil(remaining))
        try:
            artifact.wait(timeout=poll_timeout)
        except WaitTimeoutError:
            # W&B 0.16 raises this without a nested TimeoutError. It only means
            # the asynchronous upload is still pending, so no retry is consumed.
            return False
        return True

    while True:
        remaining = remaining_time()
        if wandb_retry(
            poll,
            description=description,
            max_elapsed=120.0 if remaining is None else min(120.0, remaining),
        ):
            return


def retry_wandb(function: Callable[..., T]) -> Callable[..., T]:
    """Retry a read operation, including iteration over paginated API results."""

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> T:
        return wandb_retry(lambda: function(*args, **kwargs), description=function.__name__)

    return wrapped


def wandb_client(owner: Any, overrides: Dict[str, str]) -> wandb.Api:
    """Reuse clients within a process; never inherit a parent's network client."""
    pid = os.getpid()
    if getattr(owner, "_wandb_client_pid", None) != pid:
        owner._wandb_client = wandb_retry(
            lambda: wandb.Api(overrides=overrides, timeout=30),
            description="create API client",
        )
        owner._wandb_client_pid = pid
    return owner._wandb_client


def init_wandb_run(**kwargs: Any) -> Any:
    """Retry initialization with one ID, cleaning up partially initialized runs."""
    if wandb.run is not None:
        raise RuntimeError("There is already a W&B run initialized, cannot initialize another one.")
    kwargs["id"] = kwargs.get("id") or wandb.util.generate_id()
    kwargs["resume"] = "allow"

    def initialize():
        try:
            return wandb.init(**kwargs)
        except Exception:
            if wandb.run is not None:
                try:
                    wandb.finish(exit_code=1)
                except Exception:
                    logger.exception("Unable to close partially initialized W&B run")
            raise

    return wandb_retry(
        initialize, description="initialize run", retry_if=lambda _: wandb.run is None
    )
