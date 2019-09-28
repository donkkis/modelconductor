__package__ = "modelconductor"
import abc
import pickle
import shutil
from enum import Enum
from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
from .utils import Measurement
from .utils import ModelResponse
from uuid import uuid1
from abc import abstractmethod
from .exceptions import ModelStepException

class ModelStatus(Enum):
    READY = 1
    BUSY = 2
    NOT_INITIATED = 3


class ModelHandler:

    def __init__(self, sources=None, input_keys=None, target_keys=None,
                 control_keys=None):
        """A base class for ModelHandler objects

        Args:
            sources (List[MeasurementStreamHandler]): A list of
                MeasurementStreamHandler objects associated with this
                ModelHandler instance
            input_keys (List[str]):
            target_keys (List[str]):
            control_keys (List[str]):
        """
        self.input_keys = input_keys
        self.target_keys = target_keys
        self.control_keys = control_keys
        self.sources = sources if sources is not None else []
        self.status = ModelStatus.NOT_INITIATED

    def add_source(self, source):
        """
        Args:
            source: A MeasurementStreamHandler object

        Returns:
            self.sources:
        """
        self.sources.append(source)
        return source

    def remove_source(self, source):
        self.sources.remove(source)
        return source

    def pull(self):
        """Request the next FIFO-queued data point from each source
        TODO: should be able to handle subsets of sources
        Returns: List[Measurement]

        """
        # TODO Might return None
        res = [source.give() for source in self.sources]
        return res

    def pull_batch(self, batch_size):
        """Request the next batch_size queued data points each source

        Args:
            batch_size:

        Returns: List[List[Measurement]]
        """
        res = [source.give_batch(batch_size) for source in self.sources]
        return res

    @abstractmethod
    def step_batch(self):
        """Feed-forward the model with batch of (consecutive) inputs"""
        raise NotImplementedError("Abstract method not implemented")

    @abstractmethod
    def step(self):
        """Feed-forward the model with the latest data point"""
        raise NotImplementedError("Abstract method not implemented")

    @abstractmethod
    def spawn(self):
        """Instantiate the model so that steps can be executed"""
        raise NotImplementedError("Abstract method not implemented")

    @abstractmethod
    def destroy(self):
        """Remove the model instance from the current experiment"""
        raise NotImplementedError("Abstract method not implemented")


class TrainableModelHandler(ModelHandler):

    @abstractmethod
    def fit(self):
        """Fit the model's trainable parameters"""
        raise NotImplementedError("Abstract method not implemented")

    @abstractmethod
    def fit_batch(self, measurements):
        """Train the associated model on minibatch
        Args:
            measurements: A list of measurement objects from which the
                inputs and targets will be parsed
        Returns:
        """
        raise NotImplementedError("Abstract method not implemented")

    @abstractmethod
    def step_fit_batch(self, measurements):
        raise NotImplementedError("Abstract method not implemented")


class SklearnModelHandler(TrainableModelHandler):

    def __init__(self,
                 model_filename,
                 input_keys=None,
                 target_keys=None,
                 control_keys=None,
                 sources=None):
        super().__init__(sources=sources,
                         input_keys=input_keys,
                         target_keys=target_keys,
                         control_keys=control_keys)
        with open(model_filename, 'rb') as pickle_file:
            self.model = pickle.load(pickle_file)

    def step(self, X):
        # X is dict when calling from run
        self.status = ModelStatus.BUSY
        # convert to numpy and sort columns to same order as input_keys
        # to make sure input is in format that the model expects
        if isinstance(X, dict):
            X = Measurement(X)
        for k, v in X.items():
            if v is None:
                X[k] = 0
        X = X.to_numpy(self.input_keys)
        try:
            result = list(X[0]) + list(self.model.predict(X))
            # debug
            # print("Got response from model", str(result)[0:20], "...",
            #     "[Message has been truncated]")
            self.status = ModelStatus.READY
            return [result]
        except Exception:
            raise ModelStepException(
                "Could not get a valid response from model")

    def step_batch(self, measurements):
        """Feeds a of measurements to model an returns a list of results

        Args:
            measurements: a list of Measurement objects

        Returns:
            A list of result objects
        """
        # TODO Vectorized implementation would be faster
        self.status = ModelStatus.BUSY
        results = []
        for measurement in measurements:
            results.append(self.step(measurement))
        self.status = ModelStatus.READY
        return results

    def spawn(self):
        self.status = ModelStatus.READY

    def destroy(self):
        pass

    def fit(self, X):
        # TODO need something like partial_fit from scickit-multiflow
        raise NotImplementedError("Unsupported in this ModelConductor release")

    def fit_batch(self, measurements):
        # TODO SklearnModelHandler.fit_batch
        raise NotImplementedError("Unsupported in this ModelConductor release")

    def step_fit_batch(self, measurements):
        # TODO SklearnModelHandler.step_fit_batch
        raise NotImplementedError("Unsupported in this ModelConductor release")


class FMUModelHandler(ModelHandler):

    def __init__(self, fmu_path, step_size, stop_time, timestamp_key,
                 start_time=None, target_keys=None, input_keys=None,
                 control_keys=None):
        """Loads and executes Functional Mockup Unit (FMU) modules

        Handles FMU archives adhering to the Functional Mockup Interface
        standard 2.0 as a part of a digital twin experiment. We make
        extensive use of FMPY library. The full FMI spec can be found at
        https://fmi-standard.org/

        Args:
            fmu_path (str): Path to the FMI-compliant zip-archive
            start_time (float): Value of time variable supplied to the
                FMU at the first timestep of the simulation. None (def)
                translates to 0.0
            stop_time (float): (Optional) final value of the time
                variable in the simulation. A valid FMU will report an
                error state if master tries to simulate past stop_time
            step_size (float): The default communication step size for
                the model. Can be be overridden in individual calls to
                step method to accommodate dynamical stepping
            timestamp_key: String identifier of the timestamp key
            target_keys (List(str)): Dependent variable names in the
                simulation
            input_keys (List(str)): Independent variable names in the
                simulation
            control_keys (List(str)): (Optional) Variable names e.g. for
                validating the simulation output, in meas-vs-sim style
        """
        super(FMUModelHandler, self).__init__(target_keys=target_keys,
                                              input_keys=input_keys,
                                              control_keys=control_keys)
        self.model_description = read_model_description(fmu_path)
        self.fmu_filename = fmu_path
        self.start_time = 0 if start_time is None else start_time
        # TODO parse default stop_time from model_description
        self.stop_time = stop_time
        # TODO parse default step_size from model_description
        self.step_size = step_size
        # TODO should make this work with datetimes as well as integers
        self.timestamp_key = timestamp_key
        self.unzipdir = extract(fmu_path)
        self.vrs = {}
        self._fmu = None
        self._vr_input = None
        self._vr_output = None
        self._get_value_references()

    def _get_value_references(self):
        # Get variable dictionary
        # TODO should use OrderedDict
        for var in self.model_description.modelVariables:
            if var.causality not in self.vrs.keys():
                self.vrs[var.causality] = {}
            self.vrs[var.causality][var.name] = var.valueReference

        # Get integer pointers to input variables in FMU
        self._vr_input = \
            [self.vrs["input"][k] for k, v in self.vrs["input"].items()]
        # Get integer pointers to output variables in FMU
        self._vr_output = \
            [self.vrs["output"][k] for k, v in self.vrs["output"].items()]

    def spawn(self):
        guid = self.model_description.guid
        unzipdir = self.unzipdir
        model_id = self.model_description.coSimulation.modelIdentifier
        inst_name = model_id[:8] + str(uuid1())[:8]  # Unique 16-char FMU ID

        self._fmu = FMU2Slave(guid=guid, unzipDirectory=unzipdir,
                              modelIdentifier=model_id, instanceName=inst_name)
        self._fmu.instantiate()
        self._fmu.setupExperiment(startTime=self.start_time,
                                  stopTime=self.stop_time)
        self._fmu.enterInitializationMode()
        self._fmu.exitInitializationMode()
        self.status = ModelStatus.READY

    def _build_response(self, r):
        """Attach variable names to response list"""
        keys = list(self.vrs['input'].keys()) + list(self.vrs['output'].keys())
        response = {k: v for k, v in zip(keys, r)}
        response = ModelResponse(response)
        return response

    def step(self, X, step_size=None):
        self.status = ModelStatus.BUSY

        data = [X[k] for k in self.vrs["input"].keys()]
        time = X[self.timestamp_key]

        step_size = self.step_size if step_size is None else step_size
        # set the input
        print("stepping:")  # debug
        print(data)  # debug
        self._fmu.setReal(self._vr_input, data)

        # perform one step
        self._fmu.doStep(currentCommunicationPoint=time,
                         communicationStepSize=step_size)

        # get the values for 'inputs' and 'outputs'
        response = self._fmu.getReal(self._vr_input + self._vr_output)
        response = self._build_response(response)
        print("Got response from model", response)
        self.status = ModelStatus.READY
        # TODO Fix
        return [response]

    def step_batch(self):
        raise NotImplementedError

    def destroy(self):
        self._fmu.terminate()
        self._fmu.freeInstance()
        # clean up
        shutil.rmtree(self.unzipdir)
