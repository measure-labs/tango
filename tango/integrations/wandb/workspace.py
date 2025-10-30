import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, TypeVar, Union
from urllib.parse import ParseResult

import pytz
import wandb

from tango.common.exceptions import ConfigurationError, StepStateError
from tango.common.file_lock import FileLock
from tango.common.util import exception_to_string, tango_cache_dir, utc_now_datetime
from tango.step import Step
from tango.step_cache import StepCache
from tango.step_info import StepInfo, StepState
from tango.workspace import Run, Workspace

from .step_cache import WandbStepCache
from .util import RunKind, check_environment

T = TypeVar("T")

logger = logging.getLogger(__name__)


@Workspace.register("wandb")
class WandbWorkspace(Workspace):
    """
    This is a :class:`~tango.workspace.Workspace` that tracks Tango runs in a W&B project.
    It also stores step results as W&B Artifacts via :class:`WandbStepCache`.

    Each Tango run with this workspace will generate multiple runs in your W&B project.
    There will always be a W&B run corresponding to each Tango run with the same name,
    which will contain some metadata about the Tango run. Then there will be one W&B run
    for each cacheable step that runs with a name corresponding to the name of the step.
    So if your Tango run includes 3 cacheable steps, that will result in a total of 4 new runs in W&B.

    :param project: The W&B project to use for the workspace.
    :param entity: The W&B entity (user or organization account) to use for the workspace.

    .. tip::
        Registered as a :class:`~tango.workspace.Workspace` under the name "wandb".

    .. tip::
        If you want to change the artifact kind for step result artifacts uploaded
        to W&B, add a field called ``artifact_kind`` to the ``metadata`` of
        the :class:`~tango.step.Step` class.

        This can be useful if you want model objects to be added to the model zoo.
        In that you would set ``artifact_kind = "model"``.
        For example, your config for the step would look like this:

        .. code-block::

            { type: "trainer", step_metadata: { artifact_kind: "model" }, ... }

        Or just add this to the ``METADATA`` class attribute:

        .. code-block::

            @Step.register("trainer")
            class TrainerStep(Step):
                METADATA = {"artifact_kind": "model"}
    """

    def __init__(self, project: str, entity: Optional[str] = None):
        check_environment()
        super().__init__()
        self.project = project
        self._entity = entity
        self.cache = WandbStepCache(project=self.project, entity=self.entity)
        self.steps_dir = tango_cache_dir() / "wandb_workspace"
        self.locks: Dict[Step, FileLock] = {}
        self._running_step_info: Dict[str, StepInfo] = {}

    def __getstate__(self):
        """
        We override `__getstate__()` to customize how instances of this class are pickled
        since we don't want to persist certain attributes.
        """
        out = super().__getstate__()
        out["locks"] = {}
        return out

    @property
    def wandb_client(self) -> wandb.Api:
        overrides = {"project": self.project}
        if self._entity is not None:
            overrides["entity"] = self._entity
        return wandb.Api(overrides=overrides)

    @property
    def entity(self) -> str:
        return self._entity or self.wandb_client.default_entity

    @property
    def url(self) -> str:
        return f"wandb://{self.entity}/{self.project}"

    @classmethod
    def from_parsed_url(cls, parsed_url: ParseResult) -> Workspace:
        entity = parsed_url.netloc
        project = parsed_url.path
        if project:
            project = project.strip("/")
        return cls(project=project, entity=entity)

    @property
    def step_cache(self) -> StepCache:
        return self.cache

    @property
    def wandb_project_url(self) -> str:
        """
        The URL of the W&B project this workspace uses.
        """
        app_url = self.wandb_client.client.app_url
        app_url = app_url.rstrip("/")
        return f"{app_url}/{self.entity}/{self.project}"

    def _get_unique_id(self, step_or_unique_id: Union[Step, str]) -> str:
        if isinstance(step_or_unique_id, Step):
            unique_id = step_or_unique_id.unique_id
        else:
            unique_id = step_or_unique_id
        return unique_id

    def step_dir(self, step_or_unique_id: Union[Step, str]) -> Path:
        unique_id = self._get_unique_id(step_or_unique_id)
        path = self.steps_dir / unique_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def work_dir(self, step: Step) -> Path:
        path = self.step_dir(step) / "work"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def step_info(self, step_or_unique_id: Union[Step, str]) -> StepInfo:
        unique_id = self._get_unique_id(step_or_unique_id)
        if unique_id in self._running_step_info:
            return self._running_step_info[unique_id]
        step_info = self._get_updated_step_info(
            unique_id,
            step_name=step_or_unique_id.name if isinstance(step_or_unique_id, Step) else None,
        )
        if step_info is None:
            if isinstance(step_or_unique_id, Step):
                return StepInfo.new_from_step(step_or_unique_id)
            raise KeyError(step_or_unique_id)
        return step_info

    def step_starting(self, step: Step) -> None:
        if wandb.run is not None:
            raise RuntimeError(
                "There is already a W&B run initialized, cannot initialize another one."
            )

        work_dir = self.work_dir(step)

        lock_path = self.step_dir(step) / "lock"
        lock = FileLock(lock_path, read_only_ok=True)
        lock.acquire_with_updates(desc=f"acquiring lock for '{step.name}'")
        self.locks[step] = lock

        step_info = self._get_updated_step_info(step.unique_id) or StepInfo.new_from_step(step)
        if step_info.state not in {StepState.INCOMPLETE, StepState.FAILED, StepState.UNCACHEABLE}:
            raise StepStateError(
                step,
                step_info.state,
                context="If you are certain the step is not running somewhere else, delete the lock "
                f"file at {lock_path}.",
            )

        try:
            # Initialize W&B run for the step.
            wandb.init(
                name=step_info.step_name,
                job_type=RunKind.STEP.value,
                group=step.unique_id,
                dir=str(work_dir),
                entity=self.entity,
                project=self.project,
                # For cacheable steps we can just use the step's unique ID as the W&B run ID,
                # but not for uncacheable steps since those might be ran more than once, and
                # and will need a unique W&B run ID each time.
                id=step.unique_id if step.cache_results else None,
                resume="allow" if step.cache_results else None,
                notes="\n".join(
                    [
                        f'Tango step "{step.name}"',
                        f"\N{bullet} type: {step_info.step_class_name}",
                        f"\N{bullet} ID: {step.unique_id}",
                    ]
                ),
                config={
                    "job_type": RunKind.STEP.value,
                    "_run_suite_id": self._generate_run_suite_id(),  # used for testing only
                },
            )

            assert wandb.run is not None
            logger.info(
                "Tracking '%s' step on Weights and Biases: %s/runs/%s/overview",
                step.name,
                self.wandb_project_url,
                wandb.run.id,
            )

            # "Use" all of the result artifacts for this step's dependencies in order to declare
            # those dependencies to W&B.
            for dependency in step.dependencies:
                self.cache.use_step_result_artifact(dependency)

            # Update StepInfo to mark as running.
            step_info.start_time = utc_now_datetime()
            step_info.end_time = None
            step_info.error = None
            step_info.result_location = None
            wandb.run.config.update({"step_info": step_info.to_json_dict()}, allow_val_change=True)
            self._running_step_info[step.unique_id] = step_info
        except:  # noqa: E722
            lock.release()
            del self.locks[step]
            raise

    def step_finished(self, step: Step, result: T) -> T:
        if wandb.run is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.step_finished() called outside of a W&B run. "
                f"Did you forget to call {self.__class__.__name__}.step_starting() first?"
            )

        step_info = self._running_step_info.get(step.unique_id) or self._get_updated_step_info(
            step.unique_id
        )
        if step_info is None:
            raise KeyError(step.unique_id)

        try:
            if step.cache_results:
                self.step_cache[step] = result
                if hasattr(result, "__next__"):
                    assert isinstance(result, Iterator)
                    # Caching the iterator will consume it, so we write it to the
                    # cache and then read from the cache for the return value.
                    result = self.step_cache[step]
                step_info.result_location = self.cache.get_step_result_artifact_url(step)
            else:
                # Create an empty artifact in order to build the DAG in W&B.
                self.cache.create_step_result_artifact(step)

            step_info.end_time = utc_now_datetime()
            wandb.run.config.update({"step_info": step_info.to_json_dict()}, allow_val_change=True)

            # Finalize the step's W&B run.
            wandb.finish()
        finally:
            self.locks[step].release()
            del self.locks[step]
            if step.unique_id in self._running_step_info:
                del self._running_step_info[step.unique_id]

        return result

    def step_failed(self, step: Step, e: BaseException) -> None:
        if wandb.run is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.step_failed() called outside of a W&B run. "
                f"Did you forget to call {self.__class__.__name__}.step_starting() first?"
            )

        step_info = self._running_step_info.get(step.unique_id) or self._get_updated_step_info(
            step.unique_id
        )
        if step_info is None:
            raise KeyError(step.unique_id)

        try:
            # Update StepInfo, marking the step as failed.
            if step_info.state != StepState.RUNNING:
                raise StepStateError(step, step_info.state)
            step_info.end_time = utc_now_datetime()
            step_info.error = exception_to_string(e)
            wandb.run.config.update({"step_info": step_info.to_json_dict()}, allow_val_change=True)

            # Finalize the step's W&B run.
            wandb.finish(exit_code=1)
        finally:
            self.locks[step].release()
            del self.locks[step]
            if step.unique_id in self._running_step_info:
                del self._running_step_info[step.unique_id]

    def remove_step(self, step_unique_id: str):
        """
        Removes cached step using the given unique step id
        :raises KeyError: If there is no step with the given name.
        """
        raise NotImplementedError()

    def register_run(self, targets: Iterable[Step], name: Optional[str] = None) -> Run:
        all_steps = set(targets)
        for step in targets:
            all_steps |= step.recursive_dependencies

        wandb_run_id: str
        wandb_run_name: str
        with tempfile.TemporaryDirectory() as temp_dir_name:
            with wandb.init(  # type: ignore[union-attr]
                job_type=RunKind.TANGO_RUN.value,
                entity=self.entity,
                project=self.project,
                name=name,
                dir=temp_dir_name,
                config={
                    "job_type": RunKind.TANGO_RUN.value,  # need this in the config so we can filter runs by this
                    "_run_suite_id": self._generate_run_suite_id(),  # used for testing only
                },
            ) as wandb_run:
                wandb_run_id = wandb_run.id
                wandb_run_name = wandb_run.name  # type: ignore[assignment]
                logger.info("Registering run %s with Weights and Biases", wandb_run.name)
                logger.info(
                    "View run at: %s/runs/%s/overview", self.wandb_project_url, wandb_run_id
                )

                # Collect step info for all steps.
                step_ids: Dict[str, bool] = {}
                step_name_to_info: Dict[str, Dict[str, Any]] = {}
                for step in all_steps:
                    step_info = StepInfo.new_from_step(step)
                    step_name_to_info[step.name] = {
                        k: v for k, v in step_info.to_json_dict().items() if v is not None
                    }
                    step_ids[step.unique_id] = True

                # Update config with step info.
                wandb_run.config.update({"steps": step_name_to_info, "_step_ids": step_ids})

                # Update notes.
                notes = "Tango run\n--------------"
                cacheable_steps = {step for step in all_steps if step.cache_results}
                if cacheable_steps:
                    notes += "\nCacheable steps:\n"
                    for step in sorted(cacheable_steps, key=lambda step: step.name):
                        notes += f"\N{bullet} {step.name}"
                        dependencies = step.dependencies
                        if dependencies:
                            notes += ", depends on: " + ", ".join(
                                sorted(
                                    [f"'{dep.name}'" for dep in dependencies],
                                )
                            )
                        notes += "\n  \N{rightwards arrow with hook} "
                        notes += f"{self.wandb_project_url}/runs/{step.unique_id}/overview\n"
                wandb_run.notes = notes

        return self.registered_run(wandb_run_name)

    def _generate_run_suite_id(self) -> str:
        return wandb.util.generate_id()

    def registered_runs(self) -> Dict[str, Run]:
        runs: Dict[str, Run] = {}
        matching_runs = list(
            self.wandb_client.runs(
                f"{self.entity}/{self.project}",
                filters={"config.job_type": RunKind.TANGO_RUN.value},  # type: ignore
            )
        )
        for wandb_run in matching_runs:
            runs[wandb_run.name] = self._get_run_from_wandb_run(wandb_run)
        return runs

    def registered_run(self, name: str) -> Run:
        matching_runs = list(
            self.wandb_client.runs(
                f"{self.entity}/{self.project}",
                filters={"display_name": name, "config.job_type": RunKind.TANGO_RUN.value},  # type: ignore
            )
        )
        if not matching_runs:
            raise KeyError(f"Run '{name}' not found in workspace")
        elif len(matching_runs) > 1:
            raise ValueError(f"Found more than one run named '{name}' in W&B project")
        return self._get_run_from_wandb_run(matching_runs[0])

    def _get_run_from_wandb_run(
        self,
        wandb_run: wandb.apis.public.Run,
    ) -> Run:
        config = self._wandb_config_as_dict(wandb_run)
        steps_config = self._maybe_parse_json_config_value(
            config.get("steps"), wandb_run.name, "steps"
        )
        if not isinstance(steps_config, dict):
            logger.debug(
                "Unexpected 'steps' config type '%s' for run '%s'; treating as empty.",
                type(steps_config).__name__,
                wandb_run.name,
            )
        step_name_to_info = {}
        for step_name, raw_step_info in (
            steps_config.items() if isinstance(steps_config, dict) else []
        ):
            step_info_dict = self._maybe_parse_json_config_value(
                raw_step_info, wandb_run.name, f"steps.{step_name}"
            )
            if not isinstance(step_info_dict, dict):
                logger.debug(
                    "Skipping step '%s' in run '%s' because step info is not a dict (got %s).",
                    step_name,
                    wandb_run.name,
                    type(step_info_dict).__name__,
                )
                continue
            unique_id = step_info_dict.get("unique_id") or step_info_dict.get("id")
            if unique_id is None:
                logger.debug(
                    "Step info for '%s' in run '%s' missing 'unique_id'; attempting to refresh from W&B.",
                    step_name,
                    wandb_run.name,
                )
                refreshed_step_info = self._get_updated_step_info(None, step_name=step_name)
                if refreshed_step_info is None:
                    logger.debug(
                        "Unable to refresh step info for '%s' in run '%s'; skipping step.",
                        step_name,
                        wandb_run.name,
                    )
                    continue
                step_name_to_info[step_name] = refreshed_step_info
                continue
            if "unique_id" not in step_info_dict and "id" in step_info_dict:
                step_info_dict = dict(step_info_dict)
                step_info_dict["unique_id"] = step_info_dict["id"]
                step_info_dict.pop("id", None)
            if "step_class_name" not in step_info_dict and unique_id is not None:
                step_info_dict = dict(step_info_dict)
                step_info_dict["step_class_name"] = str(unique_id).split("-", 1)[0]
            try:
                step_info = StepInfo.from_json_dict(step_info_dict)
            except ConfigurationError:
                logger.debug(
                    "Skipping step '%s' in run '%s' because step info is missing required fields.",
                    step_name,
                    wandb_run.name,
                )
                continue
            if step_info.cacheable:
                updated_step_info = self._get_updated_step_info(
                    step_info.unique_id, step_name=step_name
                )
                if updated_step_info is not None:
                    step_info = updated_step_info
            step_name_to_info[step_name] = step_info
        return Run(
            name=wandb_run.name,
            steps=step_name_to_info,
            start_date=datetime.strptime(wandb_run.created_at, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=pytz.utc
            ),
        )

    def _get_updated_step_info(
        self, step_id: Optional[str], step_name: Optional[str] = None
    ) -> Optional[StepInfo]:
        # First try to find the W&B run corresponding to the step. This will only
        # work if the step execution was started already.
        filters = {
            "config.job_type": RunKind.STEP.value,
        }
        if step_id is not None:
            filters["config.step_info.unique_id"] = step_id
        if step_name is not None:
            filters["display_name"] = step_name
        for wandb_run in self.wandb_client.runs(
            f"{self.entity}/{self.project}",
            filters=filters,  # type: ignore
        ):
            config = self._wandb_config_as_dict(wandb_run)
            step_info_data = self._maybe_parse_json_config_value(
                config.get("step_info"), wandb_run.name, "step_info"
            )
            if not isinstance(step_info_data, dict):
                logger.debug(
                    "Unexpected 'step_info' config type '%s' for run '%s'.",
                    type(step_info_data).__name__,
                    wandb_run.name,
                )
                continue
            if step_info_data.get("unique_id") is None:
                fallback_unique_id = step_info_data.get("id") or getattr(wandb_run, "id", None)
                if fallback_unique_id is not None:
                    step_info_data = dict(step_info_data)
                    step_info_data["unique_id"] = fallback_unique_id
                    step_info_data.pop("id", None)
            if step_info_data.get("step_class_name") is None:
                fallback = step_info_data.get("unique_id")
                if fallback is not None:
                    step_info_data = dict(step_info_data)
                    step_info_data["step_class_name"] = str(fallback).split("-", 1)[0]
            try:
                step_info = StepInfo.from_json_dict(step_info_data)
            except ConfigurationError:
                logger.debug(
                    "Failed to deserialize step info for run '%s'; missing required fields. Skipping.",
                    wandb_run.name,
                )
                continue
            # Might need to fix the step info the step failed and we failed to update the config.
            if step_info.start_time is None:
                step_info.start_time = datetime.strptime(
                    wandb_run.created_at, "%Y-%m-%dT%H:%M:%S"
                ).replace(tzinfo=pytz.utc)
            if wandb_run.state in {"failed", "finished"}:
                if step_info.end_time is None:
                    step_info.end_time = datetime.strptime(
                        wandb_run.heartbeatAt, "%Y-%m-%dT%H:%M:%S"
                    ).replace(tzinfo=pytz.utc)
                if wandb_run.state == "failed" and step_info.error is None:
                    step_info.error = "Exception"
            return step_info

        # If the step hasn't been started yet, we'll have to pull the step info from the
        # registered run.
        if step_id is None:
            return None
        filters = {
            "config.job_type": RunKind.TANGO_RUN.value,
            f"config._step_ids.{step_id}": True,
        }
        if step_name is not None:
            filters[f"config.steps.{step_name}.unique_id"] = step_id
        for wandb_run in self.wandb_client.runs(
            f"{self.entity}/{self.project}",
            filters=filters,  # type: ignore
        ):
            config = self._wandb_config_as_dict(wandb_run)
            steps_config = self._maybe_parse_json_config_value(
                config.get("steps"), wandb_run.name, "steps"
            )
            if not isinstance(steps_config, dict):
                logger.debug(
                    "Unexpected 'steps' config type '%s' when looking up step '%s' on run '%s'.",
                    type(steps_config).__name__,
                    step_name or step_id,
                    wandb_run.name,
                )
                continue
            if step_name is not None:
                step_info_data = self._maybe_parse_json_config_value(
                    steps_config.get(step_name), wandb_run.name, f"steps.{step_name}"
                )
            else:
                step_info_data = None
                for candidate in steps_config.values():
                    candidate_dict = self._maybe_parse_json_config_value(
                        candidate, wandb_run.name, "steps"
                    )
                    if isinstance(candidate_dict, dict) and (
                        candidate_dict.get("unique_id") or candidate_dict.get("id")
                    ) == step_id:
                        step_info_data = candidate_dict
                        break
            if not isinstance(step_info_data, dict):
                continue
            if "unique_id" not in step_info_data and "id" in step_info_data:
                step_info_data = dict(step_info_data)
                step_info_data["unique_id"] = step_info_data["id"]
                step_info_data.pop("id", None)
            if step_info_data.get("unique_id") is None and step_id is not None:
                step_info_data = dict(step_info_data)
                step_info_data["unique_id"] = step_id
            if step_info_data.get("unique_id") is None:
                logger.debug(
                    "Resolved step info for '%s' (unique_id '%s') still missing unique ID; skipping.",
                    step_name or "<unknown>",
                    step_id,
                )
                continue
            if step_info_data.get("step_class_name") is None:
                fallback = step_info_data.get("unique_id")
                if fallback is not None:
                    step_info_data = dict(step_info_data)
                    step_info_data["step_class_name"] = str(fallback).split("-", 1)[0]
            try:
                step_info = StepInfo.from_json_dict(step_info_data)
            except ConfigurationError:
                logger.debug(
                    "Step info for '%s' (unique_id '%s') missing required fields; skipping.",
                    step_name or "<unknown>",
                    step_id,
                )
                continue
            return step_info

        return None

    def _wandb_config_as_dict(self, wandb_run: wandb.apis.public.Run) -> Dict[str, Any]:
        raw_config = wandb_run.config
        if raw_config is None:
            return {}
        if isinstance(raw_config, dict):
            return raw_config
        if hasattr(raw_config, "to_dict"):
            try:
                converted = raw_config.to_dict()
            except TypeError:
                converted = None
            if isinstance(converted, dict):
                return converted
        if hasattr(raw_config, "items"):
            try:
                return {k: v for k, v in raw_config.items()}
            except TypeError:
                pass
        if isinstance(raw_config, str):
            parsed = self._maybe_parse_json_config_value(raw_config, wandb_run.name, "config")
            if isinstance(parsed, dict):
                return parsed
        try:
            return dict(raw_config)  # type: ignore[arg-type]
        except Exception:
            logger.debug("Falling back to empty config for run '%s'.", wandb_run.name)
            return {}

    def _maybe_parse_json_config_value(
        self, value: Any, run_name: str, field_name: str
    ) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                logger.debug(
                    "Failed to JSON-decode '%s' for run '%s'; leaving original value.",
                    field_name,
                    run_name,
                )
        return value
