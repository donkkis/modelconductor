__package__ = "modelconductor"
import abc
import pickle
import shutil

import numpy as np
from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
from keras import Sequential
from .utils import Measurement
from uuid import uuid1


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
        self.status = None

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
        res = [source.give_batch(batch_size=batch_size) for source in self.sources]
        return res

    @abc.abstractmethod
    def step_batch(self):
        """Feed-forward the model with batch of (consecutive) inputs"""

    @abc.abstractmethod
    def step(self):
        """Feed-forward the model with the latest data point"""

    @abc.abstractmethod
    def spawn(self):
        """Instantiate the model so that steps can be executed"""

    @abc.abstractmethod
    def destroy(self):
        """Remove the model instance from the current experiment"""


class TrainableModelHandler(ModelHandler):

    @abc.abstractmethod
    def fit(self):
        """Fit the model's trainable parameters"""
        pass

    @abc.abstractmethod
    def fit_batch(self, measurements):
        """
        Train the associated model on minibatch
        Args:
            measurements (List[Measurement]): A list of measurement objects frow which the inputs and
            targets will be parsed
        Returns:
        """
        pass

    @abc.abstractmethod
    def step_fit_batch(self, measurements):
        pass


class SklearnModelHandler(TrainableModelHandler):

    def __init__(self,
                 model_filename,
                 input_keys=None,
                 target_keys=None,
                 sources=None):
        super().__init__(sources=sources,
                         input_keys=input_keys,
                         target_keys=target_keys)
        with open(model_filename, 'rb') as pickle_file:
            self.model = pickle.load(pickle_file)

    def fit_batch(self, measurements):
        raise NotImplementedError

    def step_fit_batch(self, measurements):
        # Maintain compatibility
        self.status = "Busy"
        results = []
        for measurement in measurements:
            results.append(self.step(measurement))
        self.status = "Ready"
        return self.model, results

    def step(self, X):
        # X is dict when calling from run
        self.status = "Busy"
        # convert to numpy and sort columns to same order as input_keys
        # to make sure input is in format that the model expects
        print(X)
        if isinstance(X, dict):
            X = Measurement(X)
        for k, v in X.items():
            if v is None:
                X[k] = 0  # TODO Ugly!
        X = X.to_numpy(self.input_keys)
        # print(X)
        result = list(self.model.predict(X))
        self.status = "Ready"
        # print(result)
        return result

    def step_batch(self, X):
        raise NotImplementedError

    def fit(self, X):
        # TODO need something like partial_fit from scickit-multiflow
        raise NotImplementedError

    def spawn(self):
        self.status = "Ready"

    def destroy(self):
        raise NotImplementedError


class KerasModelHandler(TrainableModelHandler):

    def __init__(self,
                 sources=None,
                 input_keys=None,
                 target_keys=None,
                 model=None,
                 layers=None):
        """

        Args:
            layers (List[layer]): a sequential list of keras layers

            Example:

            layers = [
            Dense(32, input_shape=(784,)),
            Activation('relu'),
            Dense(10),
            Activation('softmax'),]

        """
        super().__init__(sources=sources, input_keys=input_keys, target_keys=target_keys)
        if model is None and layers is None:
            raise Exception("Keras model or layer description excpected")
        if model:
            self.model = model
        else:
            self.model = Sequential(layers)

    def step(self, X):
        """
        Predict a single output step.

        Args:
            X (Measurement): A measurement (k, v) dict object

        Returns:
            result: The resulting prediction(s) from the associated model

        """
        # X is dict when calling from run
        self.status = "Busy"
        # convert to numpy and sort columns to same order as input_keys
        # to make sure input is in format that the model expects
        if isinstance(X, dict):
            X = Measurement(X)
        X = X.to_numpy(self.input_keys)
        result = self.model.predict(X)[0][0]
        self.status = "Ready"
        # print(result)  # debug
        return result


    def spawn(self):
        self.model.compile(optimizer='rmsprop',
                           loss='mae',
                           metrics=['accuracy'])
        self.status = "Ready"
        return self.model

    def destroy(self):
        pass

    def fit(self, measurement=None, X=None, y=None):
        """
        Train the associated model on single training example

        Args:
            measurement (Measurement): A measurement object from which the inputs and targets
            will be parsed
        Returns:

        """
        self.status = "Busy"
        X = measurement.to_numpy(keys=self.input_keys)
        y = measurement.to_numpy(keys=self.target_keys)
        self.model.train_on_batch(X, y)
        self.status = "Ready"
        return self.model

    def measurements_to_numpy(self, measurements):
        """

        Args:
            measurements (List[Measurement]): A list of measurement objects frow which the inputs and
            targets will be parsed


        Returns:
            X (np.ndarray): 2d Numpy array where each row represents the feature vector
            of a single example
            y (np.ndarray): 2d Numpy array where each row repsents a target vector
        """
        X_shape = (len(measurements), len(self.input_keys))
        y_shape = (len(measurements), len(self.target_keys))

        X = np.array([meas.to_numpy(keys=self.input_keys) for meas in measurements]).reshape(X_shape)
        y = np.array([meas.to_numpy(keys=self.target_keys) for meas in measurements]).reshape(y_shape)
        return X, y

    def fit_batch(self, measurements):
        """
        Train the associated model on minibatch

        Args:
            measurements (List[Measurement]): A list of measurement objects frow which the inputs and
            targets will be parsed

        Returns:

        """
        # TODO Should integrate this into fit
        self.status = "Busy"
        # TODO fix ugly hack
        if isinstance(measurements[0], dict):
            measurements = [Measurement(measurement) for measurement in measurements]
        X, y = self.measurements_to_numpy(measurements)
        self.model.train_on_batch(X, y)
        self.status = "Ready"
        return self.model

    def step_fit_batch(self, measurements):
        self.status = "Busy"
        self.fit_batch(measurements)
        results = []
        for measurement in measurements:
            results.append(self.step(measurement))
        self.status = "Ready"
        return self.model, results


class FMUModelHandler(ModelHandler):

    def __init__(self, fmu_path, step_size, start_time=None, stop_time=None,
                 target_keys=None, input_keys=None, control_keys=None):
        """Loads and executes Functional Mockup Unit (FMU) modules

        Handles FMU archives adhering to the Functional Mockup Interface
        standard 2.0 as a part of a digital twin experiment. We make
        extensive use of the excellent FMPY library. The full FMI spec
        can be found at https://fmi-standard.org/

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
        self.start_time = start_time
        self.stop_time = stop_time
        self.step_size = step_size
        self.unzipdir = extract(fmu_path)
        self.vrs = {}
        self._fmu = None
        self._vr_input = None
        self._vr_output = None
        self._get_variable_dict()
        self._get_value_references()

    def _get_variable_dict(self):
        for var in self.model_description.modelVariables:
            if var.causality not in self.vrs.keys():
                self.vrs[var.causality] = {}
            self.vrs[var.causality][var.name] = var.valueReference

    def _get_value_references(self):
        # integer pointers to input variables in FMU
        self._vr_input = \
            [self.vrs["input"][k] for k, v in self.vrs["input"].items()]
        # integer pointers to output variables in FMU
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
        self.status = "Ready"

    def step(self, X, step_size=None):
        self.status = "Busy"

        data = [X[k] for k in self.vrs["input"].keys()]
        time = X['index']

        step_size = self.step_size if step_size is None else step_size
        # set the input
        print("stepping:")  # debug
        print(data)  # debug
        self._fmu.setReal(self.vr_input, data)

        # perform one step
        self._fmu.doStep(currentCommunicationPoint=time,
                         communicationStepSize=step_size)

        # get the values for 'inputs' and 'outputs'
        response = self._fmu.getReal(self.vr_input + self.vr_output)
        print("Got response from model", response)
        self.status = "Ready"
        # TODO Fix
        return [response]

    def step_batch(self):
        raise NotImplementedError

    def destroy(self):
        self._fmu.terminate()
        self._fmu.freeInstance()
        # clean up
        shutil.rmtree(self.unzipdir)
