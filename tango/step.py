import itertools
import logging
import random
import re
from abc import abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
    cast,
)

try:
    from typing import get_args, get_origin  # type: ignore
except ImportError:

    def get_origin(tp):  # type: ignore
        return getattr(tp, "__origin__", None)

    def get_args(tp):  # type: ignore
        return getattr(tp, "__args__", ())


from tango.common._det_hash import CustomDetHash, det_hash
from tango.common.exceptions import ConfigurationError
from tango.common.from_params import (
    infer_constructor_params,
    infer_method_params,
    pop_and_construct_arg,
)
from tango.common.logging import TangoLogger
from tango.common.params import Params
from tango.common.registrable import Registrable
from tango.format import DillFormat, Format

if TYPE_CHECKING:
    from tango.step_cache import StepCache

logger = logging.getLogger(__name__)

_version_re = re.compile("""^[a-zA-Z0-9]+$""")

T = TypeVar("T")


class Step(Registrable, Generic[T]):
    """
    This class defines one step in your experiment. To write your own step, just derive from this class
    and overwrite the :meth:`run()` method. The :meth:`run()` method must have parameters with type hints.

    ``Step.__init__()`` takes all the arguments we want to run the step with. They get passed
    to :meth:`run()` (almost) as they are. If the arguments are other instances of ``Step``, those
    will be replaced with the step's results before calling :meth:`run()`. Further, there are four special
    parameters:

    * ``step_name`` contains an optional human-readable name for the step. This name is used for
      error messages and the like, and has no consequence on the actual computation.
    * ``cache_results`` specifies whether the results of this step should be cached. If this is
      ``False``, the step is recomputed every time it is needed. If this is not set at all,
      and :attr:`CACHEABLE` is ``True``, we cache if the step is marked as :attr:`DETERMINISTIC`,
      and we don't cache otherwise.
    * ``step_format`` gives you a way to override the step's default format (which is given in :attr:`FORMAT`).
    * ``step_config`` is the original raw part of the experiment config corresponding to this step.
      This can be accessed via the :attr:`config` property within each step's :meth:`run()` method.
    """

    DETERMINISTIC: bool = True
    """This describes whether this step can be relied upon to produce the same results every time
    when given the same inputs. If this is ``False``, the step can't be cached, and neither can any
    step that depends on it."""

    CACHEABLE: Optional[bool] = None
    """This provides a direct way to turn off caching. For example, a step that reads a HuggingFace
    dataset doesn't need to be cached, because HuggingFace datasets already have their own caching
    mechanism. But it's still a deterministic step, and all following steps are allowed to cache.
    If it is ``None``, the step figures out by itself whether it should be cacheable or not."""

    VERSION: Optional[str] = None
    """This is optional, but recommended. Specifying a version gives you a way to tell Tango that
    a step has changed during development, and should now be recomputed. This doesn't invalidate
    the old results, so when you revert your code, the old cache entries will stick around and be
    picked up."""

    FORMAT: Format = DillFormat("gz")
    """This specifies the format the results of this step will be serialized in. See the documentation
    for :class:`~tango.format.Format` for details."""

    def __init__(
        self,
        step_name: Optional[str] = None,
        cache_results: Optional[bool] = None,
        step_format: Optional[Format] = None,
        step_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.logger = cast(TangoLogger, logging.getLogger(self.__class__.__name__))

        if self.VERSION is not None:
            assert _version_re.match(
                self.VERSION
            ), f"Invalid characters in version '{self.VERSION}'"
        self.kwargs = kwargs

        if step_format is None:
            self.format = self.FORMAT
            if isinstance(self.format, type):
                self.format = self.format()
        else:
            self.format = step_format

        self.unique_id_cache: Optional[str] = None
        if step_name is None:
            self.name = self.unique_id
        else:
            self.name = step_name

        if cache_results is True:
            if not self.CACHEABLE:
                raise ConfigurationError(
                    f"Step {self.name} is configured to use the cache, but it's not a cacheable step."
                )
            if not self.DETERMINISTIC:
                logger.warning(
                    f"Step {self.name} is going to be cached despite not being deterministic."
                )
            self.cache_results = True
        elif cache_results is False:
            self.cache_results = False
        elif cache_results is None:
            c = (self.DETERMINISTIC, self.CACHEABLE)
            if c == (False, None):
                self.cache_results = False
            elif c == (True, None):
                self.cache_results = True
            elif c == (False, False):
                self.cache_results = False
            elif c == (True, False):
                self.cache_results = False
            elif c == (False, True):
                logger.warning(
                    f"Step {self.name} is set to be cacheable despite not being deterministic."
                )
                self.cache_results = True
            elif c == (True, True):
                self.cache_results = True
            else:
                assert False, "Step.DETERMINISTIC or step.CACHEABLE are set to an invalid value."
        else:
            raise ConfigurationError(
                f"Step {self.name}'s cache_results parameter is set to an invalid value."
            )

        self.work_dir_for_run: Optional[
            Path
        ] = None  # This is set only while the run() method runs.

        self._config = step_config

    @classmethod
    def from_params(  # type: ignore[override]
        cls: Type["Step"],
        params: Union[Params, dict, str],
        constructor_to_call: Callable[..., "Step"] = None,
        constructor_to_inspect: Union[Callable[..., "Step"], Callable[["Step"], None]] = None,
        step_name: Optional[str] = None,
        **extras,
    ) -> "Step":
        # Why do we need a custom from_params? Step classes have a run() method that takes all the
        # parameters necessary to perform the step. The __init__() method of the step takes those
        # same parameters, but each of them could be wrapped in another Step instead of being
        # supplied directly. from_params() doesn't know anything about these shenanigans, so
        # we have to supply the necessary logic here.

        if constructor_to_call is not None:
            raise ConfigurationError(
                f"{cls.__name__}.from_params cannot be called with a constructor_to_call."
            )
        if constructor_to_inspect is not None:
            raise ConfigurationError(
                f"{cls.__name__}.from_params cannot be called with a constructor_to_inspect."
            )

        if isinstance(params, str):
            params = Params({"type": params})

        if not isinstance(params, Params):
            if isinstance(params, dict):
                params = Params(params)
            else:
                raise ConfigurationError(
                    "from_params was passed a ``params`` object that was not a ``Params``. This probably "
                    "indicates malformed parameters in a configuration file, where something that "
                    "should have been a dictionary was actually a list, or something else. "
                    f"This happened when constructing an object of type {cls}."
                )

        raw_step_config = deepcopy(params.as_dict(quiet=True))

        as_registrable = cast(Type[Registrable], cls)
        if "type" in params and params["type"] not in as_registrable.list_available():
            as_registrable.search_modules(params["type"])
        choice = params.pop_choice(
            "type", choices=as_registrable.list_available(), default_to_first_choice=True
        )
        subclass, constructor_name = as_registrable.resolve_class_name(choice)
        if not issubclass(subclass, Step):
            # This can happen if `choice` is a fully qualified name.
            raise ConfigurationError(
                f"Tried to make a Step of type {choice}, but ended up with a {subclass}."
            )

        parameters = infer_method_params(subclass, subclass.run, infer_kwargs=False)
        del parameters["self"]
        init_parameters = infer_constructor_params(subclass)
        del init_parameters["self"]
        del init_parameters["kwargs"]
        parameter_overlap = parameters.keys() & init_parameters.keys()
        assert len(parameter_overlap) <= 0, (
            f"If this assert fails it means that you wrote a Step with a run() method that takes one of the "
            f"reserved parameters ({', '.join(init_parameters.keys())})"
        )
        parameters.update(init_parameters)

        kwargs: Dict[str, Any] = {}
        accepts_kwargs = False
        for param_name, param in parameters.items():
            if param.kind == param.VAR_KEYWORD:
                # When a class takes **kwargs we store the fact that the method allows extra keys; if
                # we get extra parameters, instead of crashing, we'll just pass them as-is to the
                # constructor, and hope that you know what you're doing.
                accepts_kwargs = True
                continue

            explicitly_set = param_name in params
            constructed_arg = pop_and_construct_arg(
                subclass.__name__, param_name, param.annotation, param.default, params, **extras
            )

            # If the param wasn't explicitly set in `params` and we just ended up constructing
            # the default value for the parameter, we can just omit it.
            # Leaving it in can cause issues with **kwargs in some corner cases, where you might end up
            # with multiple values for a single parameter (e.g., the default value gives you lazy=False
            # for a dataset reader inside **kwargs, but a particular dataset reader actually hard-codes
            # lazy=True - the superclass sees both lazy=True and lazy=False in its constructor).
            if explicitly_set or constructed_arg is not param.default:
                kwargs[param_name] = constructed_arg

        if accepts_kwargs:
            kwargs.update(params)
        else:
            params.assert_empty(subclass.__name__)

        return subclass(step_name=step_name, step_config=raw_step_config, **kwargs)

    @abstractmethod
    def run(self, **kwargs) -> T:
        """
        Execute the step's action.

        This method needs to be implemented when creating a ``Step`` subclass, but
        it shouldn't be called directly. Instead, call :meth:`result()`.
        """
        raise NotImplementedError()

    def _run_with_work_dir(self, cache: "StepCache", **kwargs) -> T:
        if self.work_dir_for_run is not None:
            raise RuntimeError("You can only run a Step's run() method once at a time.")

        logger.info("Starting run for step %s of type %s", self.name, self.__class__.__name__)

        if self.DETERMINISTIC:
            random.seed(784507111)

        step_dir = cache.step_dir(self)
        self.work_dir_for_run = step_dir / "work"
        try:
            self.work_dir_for_run.mkdir(exist_ok=True, parents=True)
            return self.run(**kwargs)
        finally:
            # No cleanup, as we want to keep the directory for restarts or serialization.
            self.work_dir_for_run = None

    @property
    def work_dir(self) -> Path:
        """
        The working directory that a step can use while its ``:meth:run()`` method runs.

        This is a convenience property for you to call inside your :meth:`run()` method.

        This directory stays around across restarts. You cannot assume that it is empty when your
        step runs, but you can use it to store information that helps you restart a step if it
        got killed half-way through the last time it ran."""
        if self.work_dir_for_run is None:
            raise RuntimeError(
                "You can only call this method while the step is running with a working directory. "
                "Did you call '.run()' directly? You should only run a step with '.result()'."
            )
        return self.work_dir_for_run

    @property
    def config(self) -> Dict[str, Any]:
        """
        The configuration parameters that were used to construct the step. This can be empty
        if the step was not constructed from a configuration file.
        """
        if self._config is None:
            raise ValueError(f"No config has been assigned to this step! ('{self.name}')")
        else:
            return self._config

    def det_hash_object(self) -> Any:
        return self.unique_id

    @property
    def unique_id(self) -> str:
        """Returns the unique ID for this step.

        Unique IDs are of the shape ``$class_name-$version-$hash``, where the hash is the hash of the
        inputs for deterministic steps, and a random string of characters for non-deterministic ones."""
        if self.unique_id_cache is None:
            self.unique_id_cache = self.__class__.__name__
            if self.VERSION is not None:
                self.unique_id_cache += "-"
                self.unique_id_cache += self.VERSION

            self.unique_id_cache += "-"
            if self.DETERMINISTIC:
                self.unique_id_cache += det_hash(
                    (
                        (self.format.__class__.__module__, self.format.__class__.__qualname__),
                        self.format.VERSION,
                        self.kwargs,
                    )
                )[:32]
            else:
                self.unique_id_cache += det_hash(random.getrandbits((58 ** 32).bit_length()))[:32]

        return self.unique_id_cache

    def __hash__(self):
        return hash(self.unique_id)

    def __eq__(self, other):
        if isinstance(other, Step):
            return self.unique_id == other.unique_id
        else:
            return False

    @classmethod
    def _replace_steps_with_results(
        cls, o: Any, cache: "StepCache", non_cacheable_results: Optional[Dict["Step", Any]] = None
    ):
        if isinstance(o, Step):
            return o.result(cache, non_cacheable_results)
        if isinstance(o, WithUnresolvedSteps):
            return o.construct(cache, non_cacheable_results)
        if isinstance(o, (list, tuple, set)):
            return o.__class__(
                cls._replace_steps_with_results(i, cache, non_cacheable_results) for i in o
            )
        elif isinstance(o, dict):
            return {
                key: cls._replace_steps_with_results(value, cache, non_cacheable_results)
                for key, value in o.items()
            }
        else:
            return o

    def result(
        self,
        cache: Optional["StepCache"] = None,
        non_cacheable_results: Optional[Dict["Step", Any]] = None,
    ) -> T:
        """Returns the result of this step. If the results are cached, it returns those. Otherwise it
        runs the step and returns the result from there.

        If necessary, this method will first produce the results of all steps it depends on."""
        if non_cacheable_results and self in non_cacheable_results:
            return non_cacheable_results[self]

        if cache is None:
            from tango.step_cache import default_step_cache

            cache = default_step_cache
        if self in cache:
            return cache[self]

        kwargs = self._replace_steps_with_results(self.kwargs, cache, non_cacheable_results)
        result = self._run_with_work_dir(cache, **kwargs)
        if self.cache_results:
            cache[self] = result
            if hasattr(result, "__next__"):
                assert isinstance(result, Iterator)
                # Caching the iterator will consume it, so we write it to the cache and then read from the cache
                # for the return value.
                return cache[self]
        return result

    def ensure_result(
        self,
        cache: Optional["StepCache"] = None,
        non_cacheable_results: Optional[Dict["Step", Any]] = None,
    ) -> None:
        """This makes sure that the result of this step is in the cache. It does
        not return the result."""
        if not self.cache_results:
            raise RuntimeError(
                "It does not make sense to call ensure_result() on a step that's not cacheable."
            )

        if cache is None:
            from tango.step_cache import default_step_cache

            cache = default_step_cache
        if self in cache:
            return

        kwargs = self._replace_steps_with_results(self.kwargs, cache, non_cacheable_results)
        result = self._run_with_work_dir(cache, **kwargs)
        cache[self] = result

    def _ordered_dependencies(self) -> Iterable["Step"]:
        def dependencies_internal(o: Any) -> Iterable[Step]:
            if isinstance(o, Step):
                yield o
            elif isinstance(o, WithUnresolvedSteps):
                yield from itertools.chain(
                    dependencies_internal(o.args), dependencies_internal(o.kwargs)
                )
            elif isinstance(o, str):
                return  # Confusingly, str is an Iterable of itself, resulting in infinite recursion.
            elif isinstance(o, dict):
                yield from dependencies_internal(o.values())
            elif isinstance(o, Iterable):
                yield from itertools.chain(*(dependencies_internal(i) for i in o))
            else:
                return

        return dependencies_internal(self.kwargs.values())

    @property
    def dependencies(self) -> Set["Step"]:
        """Returns a set of steps that this step depends on.

        This does not return recursive dependencies."""
        return set(self._ordered_dependencies())

    @property
    def recursive_dependencies(self) -> Set["Step"]:
        """Returns a set of steps that this step depends on.

        This returns recursive dependencies."""

        seen = set()
        steps = list(self.dependencies)
        while len(steps) > 0:
            step = steps.pop()
            if step in seen:
                continue
            seen.add(step)
            steps.extend(step.dependencies)
        return seen


class WithUnresolvedSteps(CustomDetHash):
    """
    This is a helper class for scenarios where steps depend on other steps.

    Let's say we have two steps, :class:`ConsumeDataStep` and :class:`ProduceDataStep`. The easiest way to make
    :class:`ConsumeDataStep` depend on :class:`ProduceDataStep` is to specify ``Produce`` as one of the arguments
    to the step. This works when ``Consume`` takes the output of ``Produce`` directly, or if it takes
    it inside standard Python container, like a list, set, or dictionary.

    But what if the output of :class:`ConsumeDataStep` needs to be added to a complex, custom data
    structure? :class:`WithUnresolvedSteps` takes care of this scenario.

    For example, this works without any help:

    .. code-block:: Python

        class ProduceDataStep(Step[MyDataClass]):
            def run(self, ...) -> MyDataClass
                ...
                return MyDataClass(...)

        class ConsumeDataStep(Step):
            def run(self, input_data: MyDataClass):
                ...

        produce = ProduceDataStep()
        consume = ConsumeDataStep(input_data = produce)

    This scenario needs help:

    .. code-block:: Python

        @dataclass
        class DataWithTimestamp:
            data: MyDataClass
            timestamp: float

        class ProduceDataStep(Step[MyDataClass]):
            def run(self, ...) -> MyDataClass
                ...
                return MyDataClass(...)

        class ConsumeDataStep(Step):
            def run(self, input_data: DataWithTimestamp):
                ...

        produce = ProduceDataStep()
        consume = ConsumeDataStep(
            input_data = DataWithTimestamp(produce, time.now()))

    That does not work, because :class:`DataWithTimestamp` needs an object of type :class:`MyDataClass`, but we're
    giving it an object of type :class:`Step[MyDataClass]`. Instead, we change the last line to this:

    .. code-block:: Python

        consume = ConsumeDataStep(
            input_data = WithUnresolvedSteps(
                DataWithTimestamp, (produce, time.now()))

    :class:`WithUnresolvedSteps` will delay calling the constructor of ``DataWithTimestamp`` until
    the :meth:`run()` method runs. Tango will make sure that the results from the ``produce`` step
    are available at that time, and replaces the step in the arguments with the step's results.
    """

    def __init__(self, constructor, *args, **kwargs):
        self.constructor = constructor
        self.args = args
        self.kwargs = kwargs

    @classmethod
    def with_resolved_steps(
        cls,
        o: Any,
        step_cache: "StepCache",
        non_cacheable_results: Optional[Dict[Step, Any]] = None,
    ):
        if isinstance(o, Step):
            return o.result(step_cache, non_cacheable_results)
        elif isinstance(o, cls):
            return o.construct(step_cache, non_cacheable_results)
        elif isinstance(o, str):
            return o  # Confusingly, str is an Iterable of itself, resulting in infinite recursion.
        elif isinstance(o, dict) or isinstance(o, Params):
            return o.__class__(
                {
                    key: cls.with_resolved_steps(value, step_cache, non_cacheable_results)
                    for key, value in o.items()
                }
            )
        elif isinstance(o, (list, tuple, set)):
            return o.__class__(
                cls.with_resolved_steps(item, step_cache, non_cacheable_results) for item in o
            )
        else:
            return o

    def construct(
        self, step_cache: "StepCache", non_cacheable_results: Optional[Dict[Step, Any]] = None
    ):
        resolved_args = self.with_resolved_steps(self.args, step_cache, non_cacheable_results)
        resolved_kwargs = self.with_resolved_steps(self.kwargs, step_cache, non_cacheable_results)
        return self.constructor(*resolved_args, **resolved_kwargs)

    def det_hash_object(self) -> Any:
        return self.constructor.__qualname__, self.args, self.kwargs
