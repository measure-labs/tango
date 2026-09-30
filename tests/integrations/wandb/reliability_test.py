import json
import pickle
import random
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests
import wandb
from wandb.sdk.artifacts.exceptions import WaitTimeoutError

from tango import Step, Workspace
from tango.integrations.wandb import WandbStepCache, WandbWorkspace, reliability
from tango.integrations.wandb.util import is_missing_artifact_error


def http_error(status, message="service unavailable"):
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps({"errors": [{"message": message}]}).encode()
    return requests.HTTPError(f"HTTP {status}", response=response)


def wrapped(error):
    outer = wandb.errors.AuthenticationError("An error occurred while verifying the API key.")
    outer.__cause__ = error
    return outer


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(reliability.time, "sleep", Mock())


@pytest.fixture
def upload_clock(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(reliability.time, "monotonic", lambda: clock.now)

    def advance(seconds):
        clock.now += seconds

    reliability.time.sleep.side_effect = advance
    return clock


@pytest.mark.parametrize(
    "error,expected",
    [
        (http_error(503), True),
        (wrapped(http_error(503)), True),
        (http_error(429), True),
        (requests.ConnectionError("disconnected"), True),
        (requests.Timeout("timed out"), True),
        (
            wandb.errors.AuthenticationError("Unable to connect to server to verify API token."),
            True,
        ),
        (wrapped(http_error(401)), False),
        (http_error(403), False),
        (http_error(400), False),
        (http_error(404, "artifact 'missing' not found"), False),
        (wandb.errors.AuthenticationError("API key verification failed for host example"), False),
        (wandb.errors.CommError("artifact 'missing' not found in 'project'"), False),
        (FileNotFoundError("missing dataset ZIP"), False),
        (ValueError("invalid configuration"), False),
    ],
)
def test_retry_classification(error, expected):
    assert reliability.is_retryable(error) is expected


def test_only_the_current_missing_run_can_retry_a_404():
    error = http_error(404, "failed to find run project/new-run")
    assert reliability.is_retryable(error, run_id="new-run")
    assert not reliability.is_retryable(error, run_id="another-run")
    assert not reliability.is_retryable(error)
    assert not reliability.is_retryable(
        http_error(404, "artifact new-run not found"), run_id="new-run"
    )
    assert not reliability.is_retryable(
        http_error(404, "failed to find run project/new-run-extra"), run_id="new-run"
    )


def test_wrapped_service_failure_is_not_a_cache_miss():
    error = wandb.errors.CommError(
        "Unable to fetch artifact with name example", exc=http_error(503)
    )
    assert not is_missing_artifact_error(error)
    assert reliability.is_retryable(error)


@pytest.mark.parametrize(
    "cause",
    [
        requests.ConnectionError("disconnected"),
        requests.Timeout("timed out"),
        wandb.errors.AuthenticationError("Invalid API key"),
    ],
)
def test_wrapped_transport_or_auth_error_is_not_a_cache_miss(cause):
    error = wandb.errors.CommError("Unable to fetch artifact with name example", exc=cause)
    assert not is_missing_artifact_error(error)


def test_actual_missing_artifact_remains_a_cache_miss():
    error = wandb.errors.CommError(
        "Unable to fetch artifact with name example", exc=http_error(404, "artifact not found")
    )
    assert is_missing_artifact_error(error)
    assert not reliability.is_retryable(error)


def test_retry_recovers_and_has_an_attempt_limit():
    operation = Mock(side_effect=[wrapped(http_error(503)), 42])
    assert reliability.wandb_retry(operation, description="save") == 42
    assert operation.call_count == 2
    failure = http_error(503)
    operation = Mock(side_effect=failure)
    with pytest.raises(requests.HTTPError) as caught:
        reliability.wandb_retry(operation, description="save", max_attempts=3)
    assert caught.value is failure
    assert operation.call_count == 3


def test_retry_honors_elapsed_budget_and_retry_after():
    error = http_error(429)
    error.response.headers["Retry-After"] = "7"
    operation = Mock(side_effect=[error, 42])
    assert reliability.wandb_retry(operation, description="read") == 42
    reliability.time.sleep.assert_called_once_with(7)
    reliability.time.sleep.reset_mock()
    operation = Mock(side_effect=error)
    with pytest.raises(requests.HTTPError):
        reliability.wandb_retry(operation, description="read", max_elapsed=5)
    assert operation.call_count == 1
    reliability.time.sleep.assert_not_called()


def test_retry_jitter_preserves_global_random_state():
    state = random.getstate()
    operation = Mock(side_effect=[http_error(503), 42])
    assert reliability.wandb_retry(operation, description="read") == 42
    assert random.getstate() == state
    assert 2 <= reliability.time.sleep.call_args.args[0] <= 3


def test_cause_cycles_do_not_hang():
    error = ValueError("bad config")
    error.__cause__ = error
    assert not reliability.is_retryable(error)


def test_clients_are_reused_per_process_and_not_pickled(monkeypatch, tmp_path):
    monkeypatch.setattr("tango.integrations.wandb.step_cache.tango_cache_dir", lambda: tmp_path)
    monkeypatch.setattr("tango.integrations.wandb.workspace.tango_cache_dir", lambda: tmp_path)
    api = Mock(
        return_value=SimpleNamespace(client=SimpleNamespace(app_url="https://private.example/"))
    )
    monkeypatch.setattr(wandb, "Api", api)
    for owner in (WandbStepCache("project", "entity"), WandbWorkspace("project", "entity")):
        api.reset_mock()
        first = owner.wandb_client
        assert owner.wandb_client is first
        assert owner.wandb_project_url == "https://private.example/entity/project"
        assert owner.wandb_project_url == "https://private.example/entity/project"
        assert api.call_count == 1
        restored = pickle.loads(pickle.dumps(owner))
        assert restored.project == "project"
        assert restored.entity == "entity"
        assert "_wandb_client" not in restored.__dict__
        restored.wandb_client
        assert api.call_count == 2
        owner._wandb_client_pid = -1
        owner.wandb_client
        assert api.call_count == 3


def test_initialization_retry_keeps_run_id_and_closes_partial_run(monkeypatch):
    monkeypatch.setattr(wandb, "run", None)
    seen_ids = []
    run = Mock()

    def initialize(**kwargs):
        seen_ids.append(kwargs["id"])
        wandb.run = run
        if len(seen_ids) == 1:
            raise wrapped(http_error(503))
        return run

    finish = Mock(side_effect=lambda **_: setattr(wandb, "run", None))
    monkeypatch.setattr(wandb, "init", initialize)
    monkeypatch.setattr(wandb, "finish", finish)
    assert reliability.init_wandb_run(project="project") is run
    assert len(seen_ids) == 2 and seen_ids[0] == seen_ids[1]
    finish.assert_called_once_with(exit_code=1)


class ArtifactStep(Step):
    CACHEABLE = True

    def run(self) -> int:
        return 1


def test_artifact_upload_retries_each_phase_without_recomputing(monkeypatch, tmp_path):
    monkeypatch.setattr("tango.integrations.wandb.step_cache.tango_cache_dir", lambda: tmp_path)
    artifact = Mock(aliases=[], _save_handle=None)
    artifact.save.side_effect = [wrapped(http_error(503)), None, http_error(503), None]
    artifact.wait.side_effect = [http_error(503), artifact, artifact]
    factory = Mock(return_value=artifact)
    monkeypatch.setattr(wandb, "Artifact", factory)
    step = ArtifactStep()
    cache = WandbStepCache("project", "entity")
    cache[step] = 42
    assert factory.call_count == 1
    assert artifact.add_dir.call_count == 1
    assert artifact.save.call_count == 4
    assert artifact.wait.call_count == 3
    assert artifact.aliases == [step.unique_id]
    assert cache._metadata_path(step).is_file()


def test_pending_upload_is_not_submitted_twice(monkeypatch, tmp_path):
    monkeypatch.setattr("tango.integrations.wandb.step_cache.tango_cache_dir", lambda: tmp_path)
    artifact = Mock(aliases=[], _save_handle=None)

    def save():
        if artifact._save_handle is None:
            artifact._save_handle = object()
            raise http_error(503)

    artifact.save.side_effect = save
    monkeypatch.setattr(wandb, "Artifact", Mock(return_value=artifact))
    WandbStepCache("project", "entity")._upload_step_remote(ArtifactStep())
    # One initial submission and one alias update; the retry resumes at wait().
    assert artifact.save.call_count == 2


@pytest.mark.parametrize("upload_timeout", [None, 600])
@pytest.mark.parametrize("nested_timeout", [False, True])
def test_slow_upload_completes_without_consuming_retries(
    monkeypatch, tmp_path, upload_clock, upload_timeout, nested_timeout
):
    monkeypatch.setattr("tango.integrations.wandb.step_cache.tango_cache_dir", lambda: tmp_path)
    artifact = Mock(aliases=[], _save_handle=None)

    def wait(timeout):
        if upload_clock.now < 300:
            upload_clock.now += min(timeout, 300 - upload_clock.now)
            if upload_clock.now < 300:
                error = WaitTimeoutError("still uploading")
                if nested_timeout:
                    error.__cause__ = TimeoutError("not ready")
                raise error
        return artifact

    artifact.wait.side_effect = wait
    monkeypatch.setattr(wandb, "Artifact", Mock(return_value=artifact))
    step = ArtifactStep()
    cache = WandbStepCache("project", "entity", upload_timeout=upload_timeout)
    cache[step] = 42
    assert artifact.aliases == [step.unique_id]
    assert artifact.save.call_count == 2
    assert cache._metadata_path(step).is_file()
    assert upload_clock.now == 300
    reliability.time.sleep.assert_not_called()


def test_slow_upload_can_retry_a_subsequent_transport_failure(upload_clock):
    artifact = Mock()
    failed = False

    def wait(timeout):
        nonlocal failed
        if upload_clock.now < 180:
            upload_clock.now += timeout
            raise WaitTimeoutError("still uploading")
        if not failed:
            failed = True
            raise http_error(503)
        return artifact

    artifact.wait.side_effect = wait
    reliability.wait_for_artifact(artifact, description="upload")
    assert artifact.wait.call_count == 8
    reliability.time.sleep.assert_called_once()


def test_pending_upload_stops_at_configured_deadline(monkeypatch, tmp_path, upload_clock):
    monkeypatch.setattr("tango.integrations.wandb.step_cache.tango_cache_dir", lambda: tmp_path)
    artifact = Mock(aliases=[], _save_handle=None)

    def wait(timeout):
        upload_clock.now += timeout
        raise WaitTimeoutError("still uploading")

    artifact.wait.side_effect = wait
    monkeypatch.setattr(wandb, "Artifact", Mock(return_value=artifact))
    step = ArtifactStep()
    cache = WandbStepCache("project", "entity", upload_timeout=65)
    with pytest.raises(WaitTimeoutError, match="65 seconds"):
        cache[step] = 42
    assert upload_clock.now == 65
    assert [call.kwargs["timeout"] for call in artifact.wait.call_args_list] == [30, 30, 5]
    artifact.save.assert_called_once()
    assert artifact.aliases == []
    assert not cache._metadata_path(step).exists()
    reliability.time.sleep.assert_not_called()


def test_upload_wait_transport_failures_remain_bounded(upload_clock):
    artifact = Mock()
    error = http_error(503)
    artifact.wait.side_effect = error
    with pytest.raises(requests.HTTPError) as caught:
        reliability.wait_for_artifact(artifact, description="upload")
    assert caught.value is error
    assert artifact.wait.call_count == 6
    assert upload_clock.now < 120


def test_upload_deadline_does_not_restart_after_transport_retry(monkeypatch, upload_clock):
    monkeypatch.setattr(reliability._retry_random, "uniform", lambda *_: 0)
    artifact = Mock()

    def wait(timeout):
        if artifact.wait.call_count == 1:
            upload_clock.now += 10
            raise http_error(503)
        upload_clock.now += timeout
        raise WaitTimeoutError("still uploading")

    artifact.wait.side_effect = wait
    with pytest.raises(WaitTimeoutError, match="65 seconds"):
        reliability.wait_for_artifact(artifact, description="upload", timeout=65)
    assert upload_clock.now == 65
    assert [call.kwargs["timeout"] for call in artifact.wait.call_args_list] == [30, 30, 23]


@pytest.mark.parametrize("owner_class", [WandbStepCache, WandbWorkspace])
def test_upload_timeout_survives_pickling(owner_class, monkeypatch, tmp_path):
    monkeypatch.setattr("tango.integrations.wandb.step_cache.tango_cache_dir", lambda: tmp_path)
    monkeypatch.setattr("tango.integrations.wandb.workspace.tango_cache_dir", lambda: tmp_path)
    owner = owner_class("project", "entity", upload_timeout=600)
    restored = pickle.loads(pickle.dumps(owner))
    cache = restored.cache if isinstance(restored, WandbWorkspace) else restored
    assert cache.upload_timeout == 600


@pytest.mark.parametrize("upload_timeout", [None, 1, 600])
def test_upload_timeout_survives_workspace_url_round_trip(upload_timeout, monkeypatch, tmp_path):
    monkeypatch.setattr("tango.integrations.wandb.step_cache.tango_cache_dir", lambda: tmp_path)
    monkeypatch.setattr("tango.integrations.wandb.workspace.tango_cache_dir", lambda: tmp_path)
    workspace = WandbWorkspace("project", "entity", upload_timeout=upload_timeout)

    # Executors reconstruct worker workspaces from this URL.
    restored = Workspace.from_url(workspace.url)

    assert isinstance(restored, WandbWorkspace)
    assert restored.entity == workspace.entity
    assert restored.project == workspace.project
    assert restored.cache.upload_timeout == upload_timeout
    expected_url = "wandb://entity/project"
    if upload_timeout is not None:
        expected_url += f"?upload_timeout={upload_timeout}"
    assert workspace.url == restored.url == expected_url


@pytest.mark.parametrize("upload_timeout", ["0", "-1", "invalid", ""])
def test_workspace_url_rejects_invalid_upload_timeout(upload_timeout, monkeypatch, tmp_path):
    monkeypatch.setattr("tango.integrations.wandb.step_cache.tango_cache_dir", lambda: tmp_path)
    monkeypatch.setattr("tango.integrations.wandb.workspace.tango_cache_dir", lambda: tmp_path)
    with pytest.raises(ValueError):
        Workspace.from_url(f"wandb://entity/project?upload_timeout={upload_timeout}")


@pytest.mark.parametrize("upload_timeout", [0, -1])
@pytest.mark.parametrize("owner_class", [WandbStepCache, WandbWorkspace])
def test_upload_timeout_must_be_positive(owner_class, upload_timeout):
    with pytest.raises(ValueError, match="upload_timeout must be positive"):
        owner_class("project", "entity", upload_timeout=upload_timeout)


def test_dependency_run_404_recovers(monkeypatch, tmp_path):
    monkeypatch.setattr("tango.integrations.wandb.step_cache.tango_cache_dir", lambda: tmp_path)
    run = Mock(id="new-run")
    run.use_artifact.side_effect = [http_error(404, "failed to find run project/new-run"), None]
    monkeypatch.setattr(wandb, "run", run)
    WandbStepCache("project", "entity").use_step_result_artifact(ArtifactStep())
    assert run.use_artifact.call_count == 2


def test_download_outage_is_not_reported_as_missing_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr("tango.integrations.wandb.step_cache.tango_cache_dir", lambda: tmp_path)
    error = wrapped(http_error(503))
    artifact = Mock()
    artifact.download.side_effect = error
    with pytest.raises(wandb.errors.AuthenticationError) as caught:
        WandbStepCache("project", "entity")._download_step_remote(artifact, tmp_path)
    assert caught.value is error


def test_sdk_artifact_wait_timeout_keeps_waiting(monkeypatch, tmp_path, upload_clock):
    # Exercise both the standalone timeout in 0.16 and the chained one in 0.22.
    monkeypatch.setenv("WANDB_CACHE_DIR", str(tmp_path))
    artifact = wandb.Artifact("test-result", type="step_result")
    legacy = hasattr(artifact, "_save_future")
    response = SimpleNamespace(error_message="", artifact_id="uploaded-id")
    result = SimpleNamespace(response=SimpleNamespace(log_artifact_response=response))

    def wait(timeout):
        upload_clock.now += timeout
        if upload_clock.now < 300:
            if legacy:
                return None
            raise TimeoutError("not ready")
        return result

    if legacy:
        artifact._save_future = Mock()
        artifact._save_future.get.side_effect = wait
    else:
        artifact._save_handle = Mock()
        artifact._save_handle.wait_or.side_effect = wait
    monkeypatch.setattr(artifact, "_populate_after_save", Mock())
    reliability.wait_for_artifact(artifact, description="upload")
    artifact._populate_after_save.assert_called_once_with("uploaded-id")
    assert upload_clock.now == 300
    reliability.time.sleep.assert_not_called()


@pytest.mark.parametrize("method", ["registered_runs", "registered_run", "_get_updated_step_info"])
def test_reused_workspace_client_refreshes_run_queries(method, monkeypatch, tmp_path):
    monkeypatch.setattr("tango.integrations.wandb.workspace.tango_cache_dir", lambda: tmp_path)
    monkeypatch.setattr("tango.integrations.wandb.step_cache.tango_cache_dir", lambda: tmp_path)

    class CachedAPI:
        cached = None
        calls = 0

        def flush(self):
            self.cached = None

        def runs(self, *args, **kwargs):
            if self.cached is None:
                self.calls += 1
                self.cached = []
            return self.cached

    api = CachedAPI()
    factory = Mock(return_value=api)
    monkeypatch.setattr(wandb, "Api", factory)
    workspace = WandbWorkspace("project", "entity")
    for _ in range(2):
        if method == "registered_run":
            with pytest.raises(KeyError):
                workspace.registered_run("missing")
        elif method == "registered_runs":
            workspace.registered_runs()
        else:
            workspace._get_updated_step_info("step-id")
    assert api.calls == 2
    factory.assert_called_once()


def test_api_creation_recovers_from_wrapped_outage(monkeypatch):
    client = object()
    factory = Mock(side_effect=[wrapped(http_error(503)), client])
    monkeypatch.setattr(wandb, "Api", factory)
    owner = SimpleNamespace()
    assert reliability.wandb_client(owner, {"project": "test"}) is client
    assert reliability.wandb_client(owner, {"project": "test"}) is client
    assert factory.call_count == 2


def test_authentication_failure_is_not_retried(monkeypatch):
    failure = wrapped(http_error(401))
    operation = Mock(side_effect=failure)
    with pytest.raises(wandb.errors.AuthenticationError) as caught:
        reliability.wandb_retry(operation, description="authenticate")
    assert caught.value is failure
    operation.assert_called_once()


def test_exhausted_artifact_upload_does_not_publish_alias_or_local_cache(monkeypatch, tmp_path):
    monkeypatch.setattr("tango.integrations.wandb.step_cache.tango_cache_dir", lambda: tmp_path)
    artifact = Mock(aliases=[], _save_handle=None)
    artifact.save.side_effect = wrapped(http_error(503))
    monkeypatch.setattr(wandb, "Artifact", Mock(return_value=artifact))
    step = ArtifactStep()
    cache = WandbStepCache("project", "entity")
    with pytest.raises(wandb.errors.AuthenticationError):
        cache[step] = 42
    assert artifact.save.call_count == 6
    assert artifact.aliases == []
    artifact.wait.assert_not_called()
    assert not cache._metadata_path(step).exists()
