from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import wandb

from tango import Step
from tango.common.testing.steps import RandomStringStep
from tango.integrations.wandb import WandbWorkspace, reliability
from tango.step_info import StepState
from tango.workspaces.memory_workspace import MemoryWorkspace

from .reliability_test import http_error, wrapped


class FailingStep(Step):
    CACHEABLE = True

    def run(self) -> int:
        raise FileNotFoundError("missing prepared dataset ZIP")


class SuccessfulStep(Step):
    CACHEABLE = True

    def run(self) -> int:
        return 42


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    monkeypatch.setattr("tango.integrations.wandb.workspace.tango_cache_dir", lambda: tmp_path)
    monkeypatch.setattr("tango.integrations.wandb.step_cache.tango_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(reliability.time, "sleep", Mock())
    monkeypatch.setattr(wandb, "run", None)
    api = SimpleNamespace(client=SimpleNamespace(app_url="https://wandb.example/"))
    monkeypatch.setattr(wandb, "Api", Mock(return_value=api))

    def initialize(**kwargs):
        run = Mock(id=kwargs["id"])
        wandb.run = run
        return run

    monkeypatch.setattr(wandb, "init", Mock(side_effect=initialize))
    monkeypatch.setattr(wandb, "finish", Mock(side_effect=lambda **_: setattr(wandb, "run", None)))
    workspace = WandbWorkspace("project", "entity")
    monkeypatch.setattr(workspace, "_get_updated_step_info", Mock(return_value=None))
    monkeypatch.setattr(workspace.cache, "_step_result_remote", Mock(return_value=None))
    monkeypatch.setattr(workspace.cache, "_upload_step_remote", Mock())
    return workspace


def assert_released(workspace):
    assert not workspace.locks
    assert not workspace._running_step_info
    assert wandb.run is None


def test_startup_retry_preserves_deterministic_step_result(workspace, monkeypatch):
    step = RandomStringStep(cache_results=False)
    expected = step.result(workspace)
    initialize = wandb.init.side_effect
    attempts = 0

    def initialize_with_retry(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise http_error(503)
        return initialize(**kwargs)

    monkeypatch.setattr(wandb, "init", Mock(side_effect=initialize_with_retry))
    assert step.result(workspace) == expected
    assert attempts == 2
    assert_released(workspace)


@pytest.mark.parametrize("cache_results", [True, False])
def test_step_failure_preserves_original_error_and_allows_next_step(workspace, cache_results):
    step = FailingStep(cache_results=cache_results)
    with pytest.raises(FileNotFoundError, match="missing prepared dataset ZIP"):
        step.result(workspace)
    assert_released(workspace)
    wandb.finish.assert_called_once_with(exit_code=1)

    # This used to fail with "There is already a W&B run initialized".
    assert SuccessfulStep(cache_results=False).result(workspace) == 42
    assert_released(workspace)


def test_upload_failure_retains_local_state_until_failure_reporting(workspace):
    step = SuccessfulStep()
    error = wrapped(http_error(503))
    workspace.cache._upload_step_remote.side_effect = error
    workspace.step_starting(step)
    info = workspace._running_step_info[step.unique_id]
    run = wandb.run
    lock = workspace.locks[step]

    with pytest.raises(wandb.errors.AuthenticationError) as caught:
        workspace.step_finished(step, 42)
    assert caught.value is error
    assert workspace._running_step_info[step.unique_id] is info
    assert lock.is_locked
    # Failure cleanup must not make a second, fallible query to find its state.
    workspace._get_updated_step_info.side_effect = AssertionError("unnecessary API query")
    workspace.step_failed(step, error)

    assert info.state == StepState.FAILED
    assert "AuthenticationError" in info.error
    assert not lock.is_locked
    assert_released(workspace)
    wandb.finish.assert_called_once_with(exit_code=1)
    assert run.config.update.call_args.args[0]["step_info"]["error"]
    assert not workspace.cache._metadata_path(step).exists()


def test_failure_reporting_outage_does_not_mask_step_error(workspace, monkeypatch):
    initialize = wandb.init.side_effect

    def initialize_with_failing_config(**kwargs):
        run = initialize(**kwargs)
        # The startup config succeeds, but all failure updates fail.
        run.config.update.side_effect = [None] + [http_error(503)] * 6
        return run

    monkeypatch.setattr(wandb, "init", Mock(side_effect=initialize_with_failing_config))
    with pytest.raises(FileNotFoundError, match="missing prepared dataset ZIP"):
        FailingStep(cache_results=False).result(workspace)
    assert_released(workspace)
    wandb.finish.assert_called_once_with(exit_code=1)


def test_startup_dependency_failure_cleans_up_run_and_lock(workspace, monkeypatch):
    dependency = SuccessfulStep()
    step = SuccessfulStep(extra_dependencies=[dependency])
    error = http_error(404, f"failed to find run project/{step.unique_id}")
    initialize = wandb.init.side_effect

    def initialize_with_missing_run(**kwargs):
        run = initialize(**kwargs)
        run.use_artifact.side_effect = error
        return run

    monkeypatch.setattr(wandb, "init", Mock(side_effect=initialize_with_missing_run))
    with pytest.raises(type(error)) as caught:
        workspace.step_starting(step)
    assert caught.value is error
    assert_released(workspace)
    wandb.finish.assert_called_once_with(exit_code=1)


def test_lookup_failure_releases_lock_without_starting_a_run(workspace):
    error = wrapped(http_error(503))
    workspace._get_updated_step_info.side_effect = error
    with pytest.raises(wandb.errors.AuthenticationError) as caught:
        workspace.step_starting(SuccessfulStep())
    assert caught.value is error
    assert_released(workspace)
    wandb.init.assert_not_called()
    wandb.finish.assert_not_called()


def test_cleanup_is_idempotent_and_does_not_close_unrelated_run(workspace, monkeypatch):
    step = SuccessfulStep()
    workspace.step_starting(step)
    workspace.step_failed(step, ValueError("bad data"))
    monkeypatch.setattr(wandb, "run", Mock())
    workspace.step_failed(step, ValueError("bad data"))
    wandb.finish.assert_called_once_with(exit_code=1)
    assert wandb.run is not None


def test_unexpected_workspace_reporting_error_does_not_mask_original(monkeypatch):
    workspace = MemoryWorkspace()
    monkeypatch.setattr(workspace, "step_failed", Mock(side_effect=KeyError("missing state")))
    with pytest.raises(FileNotFoundError, match="missing prepared dataset ZIP"):
        FailingStep(cache_results=False).result(workspace)


def test_failed_init_cleanup_preserves_original_error_without_reinitializing(monkeypatch):
    monkeypatch.setattr(wandb, "run", None)
    error = wrapped(http_error(503))

    def initialize(**kwargs):
        wandb.run = Mock()
        raise error

    init = Mock(side_effect=initialize)
    monkeypatch.setattr(wandb, "init", init)
    monkeypatch.setattr(wandb, "finish", Mock(side_effect=RuntimeError("cleanup failed")))
    with pytest.raises(wandb.errors.AuthenticationError) as caught:
        reliability.init_wandb_run(project="project")
    assert caught.value is error
    init.assert_called_once()
