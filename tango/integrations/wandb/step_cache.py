import logging
from typing import Any, Optional, Union

import wandb
from wandb.errors import Error as WandbError

from tango.common.aliases import PathOrStr
from tango.common.util import make_safe_filename, tango_cache_dir
from tango.step import Step
from tango.step_cache import StepCache
from tango.step_caches.remote_step_cache import RemoteNotFoundError, RemoteStepCache
from tango.step_info import StepInfo

from .reliability import retry_wandb, wait_for_artifact, wandb_client, wandb_retry
from .util import ArtifactKind, check_environment, is_missing_artifact_error

logger = logging.getLogger(__name__)


@StepCache.register("wandb")
class WandbStepCache(RemoteStepCache):
    """
    This is a :class:`~tango.step_cache.StepCache` that's used by :class:`WandbWorkspace`.
    It stores the results of steps on W&B as Artifacts.

    It also keeps a limited in-memory cache as well as a local backup on disk, so fetching a
    step's resulting subsequent times should be fast.

    :param project: The W&B project to use.
    :param entity: The W&B entity (user or organization account) to use.
    :param upload_timeout: Maximum seconds to wait for each artifact upload to complete.
        Defaults to ``None`` (no completion deadline). Transport failures have their own
        bounded retry budget.

    .. tip::
        Registered as :class:`~tango.step_cache.StepCache` under the name "wandb".
    """

    def __init__(self, project: str, entity: str, upload_timeout: Optional[int] = None):
        if upload_timeout is not None and upload_timeout <= 0:
            raise ValueError("upload_timeout must be positive or None")
        check_environment()
        super().__init__(
            tango_cache_dir()
            / "wandb_cache"
            / make_safe_filename(entity)
            / make_safe_filename(project)
        )
        self.project = project
        self.entity = entity
        self.upload_timeout = upload_timeout

    @property
    def wandb_client(self) -> wandb.Api:
        return wandb_client(self, {"entity": self.entity, "project": self.project})

    def __getstate__(self):
        state = super().__getstate__()
        state.update(entity=self.entity, project=self.project, upload_timeout=self.upload_timeout)
        return state

    @property
    def client(self):
        """
        To maintain compatibility
        """
        return self.wandb_client

    @property
    def wandb_project_url(self) -> str:
        """
        The URL of the W&B project this workspace uses.
        """
        app_url = self.wandb_client.client.app_url
        app_url = app_url.rstrip("/")
        return f"{app_url}/{self.entity}/{self.project}"

    def _step_artifact_name(self, step: Union[Step, StepInfo]) -> str:
        if isinstance(step, Step):
            return step.class_name
        else:
            return step.step_class_name

    @retry_wandb
    def _step_result_remote(  # type: ignore
        self, step: Union[Step, StepInfo]
    ) -> Optional[wandb.Artifact]:
        artifact_kind = (step.metadata or {}).get("artifact_kind", ArtifactKind.STEP_RESULT.value)
        try:
            return self.wandb_client.artifact(
                f"{self.entity}/{self.project}/{self._step_artifact_name(step)}:{step.unique_id}",
                type=artifact_kind,
            )
        except WandbError as exc:
            if is_missing_artifact_error(exc):
                return None
            else:
                raise

    def create_step_result_artifact(self, step: Step, objects_dir: Optional[PathOrStr] = None):
        self._upload_step_remote(step, objects_dir)

    @retry_wandb
    def get_step_result_artifact(self, step: Union[Step, StepInfo]) -> Optional[wandb.Artifact]:
        artifact_kind = (step.metadata or {}).get("artifact_kind", ArtifactKind.STEP_RESULT.value)
        try:
            return self.wandb_client.artifact(
                f"{self.entity}/{self.project}/{self._step_artifact_name(step)}:{step.unique_id}",
                type=artifact_kind,
            )
        except WandbError as exc:
            if is_missing_artifact_error(exc):
                return None
            else:
                raise

    def _upload_step_remote(self, step: Step, objects_dir: Optional[PathOrStr] = None) -> Any:
        """
        Create an artifact for the result of a step.
        """
        artifact_kind = (step.metadata or {}).get("artifact_kind", ArtifactKind.STEP_RESULT.value)
        artifact = wandb.Artifact(self._step_artifact_name(step), type=artifact_kind)

        # Add files
        if objects_dir is not None:
            artifact.add_dir(str(objects_dir))

        # Log/persist the artifact to W&B.
        def save_pending_artifact():
            # A wait/population failure must not submit the payload again. W&B
            # 0.22.2 keeps this handle after submitting an asynchronous upload.
            if getattr(artifact, "_save_handle", None) is None:
                artifact.save()

        wandb_retry(save_pending_artifact, description=f"upload artifact for {step.unique_id}")
        wait_for_artifact(
            artifact, description=f"wait for {step.unique_id}", timeout=self.upload_timeout
        )

        # Add an alias for the step's unique ID.
        # Only after we've logged the artifact can we add an alias.
        if step.unique_id not in artifact.aliases:
            artifact.aliases.append(step.unique_id)
        wandb_retry(artifact.save, description=f"save alias for {step.unique_id}")
        wait_for_artifact(
            artifact, description=f"wait for alias {step.unique_id}", timeout=self.upload_timeout
        )

    def get_step_result_artifact_url(self, step: Union[Step, StepInfo]) -> str:
        artifact_kind = (step.metadata or {}).get("artifact_kind", ArtifactKind.STEP_RESULT.value)
        return (
            f"{self.wandb_project_url}/artifacts/{artifact_kind}"
            f"/{self._step_artifact_name(step)}/{step.unique_id}"
        )

    def use_step_result_artifact(self, step: Union[Step, StepInfo]) -> None:
        """
        "Use" the artifact corresponding to the result of a step.
        """
        if wandb.run is None:
            raise RuntimeError("This can only be called from within a W&B run")
        run = wandb.run
        wandb_retry(
            lambda: run.use_artifact(
                f"{self.entity}/{self.project}/{self._step_artifact_name(step)}:{step.unique_id}"
            ),
            description=f"attach dependency {step.unique_id}",
            run_id=run.id,
        )

    def _download_step_remote(self, step_result, target_dir: PathOrStr):
        try:
            wandb_retry(
                lambda: step_result.download(root=target_dir), description="download artifact"
            )
        except WandbError as error:
            if is_missing_artifact_error(error):
                raise RemoteNotFoundError() from error
            raise

    @retry_wandb
    def __len__(self) -> int:
        self.wandb_client.flush()
        completed_cacheable_step_runs = self.wandb_client.runs(
            f"{self.entity}/{self.project}",
            filters={  # type: ignore
                "config.job_type": "step",
                "config.cacheable": True,
                "state": "finished",
            },
        )
        return len(list(completed_cacheable_step_runs))
