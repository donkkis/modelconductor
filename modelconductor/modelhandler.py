__package__ = "modelconductor"
import abc
import pickle
import shutil

import numpy as np
from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
from keras import Sequential
from .utils import Measurement


class ModelHandler:

    def __init__(self, sources=None, input_keys=None, target_keys=None):
        """
        Args:
            sources (List[MeasurementStreamHandler]): A list of MeasurementStreamHandler objects
            associated with this ModelHandler instance
        """
        self.input_keys = input_keys
        self.target_keys = target_keys
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
        """
        Request the next FIFO-queued datapoint from each of the sources
        TODO: should be able to handle subsets of sources
        Returns: List[Measurement]

        """
        # TODO Might return None
        res = [source.give() for source in self.sources]
        return res

    def pull_batch(self, batch_size):
        """

        Args:
            batch_size: Request the next batch_size FIFO-queued datapoints
            from each of the sources

        Returns: List[List[Measurement]]

        """
        res = [source.give_batch(batch_size=batch_size) for source in self.sources]
        return res

    @abc.abstractmethod
    def step_batch(self):
        """Feed-forward the associated model with a batch of (consecutive) inputs"""

    @abc.abstractmethod
    def step(self):
        """Feed-forward the associated model with the latest datapoint and return the response"""

    @abc.abstractmethod
    def spawn(self):
        """Instantiate the associated model so that steps can be executed"""

    @abc.abstractmethod
    def destroy(self):
        """Remove the model instance from the current configuration and delete self"""


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
    """Loads and executes FMU binary modules
    """

    def __init__(self, fmu_filename, start_time, threshold, stop_time, step_size, target_keys=None, input_keys=None,):
        super(FMUModelHandler, self).__init__(target_keys=target_keys, input_keys=input_keys)
        self.model_description = read_model_description(fmu_filename)
        self.fmu_filename = fmu_filename
        self.start_time = start_time
        self.threshold = threshold
        self.stop_time = stop_time
        self.step_size = step_size
        self.unzipdir = extract(fmu_filename)
        self.fmu = None
        self.vrs = {}

        for variable in self.model_description.modelVariables:
            self.vrs[variable.name] = variable.valueReference

    def spawn(self):

        self.fmu = FMU2Slave(
            guid=self.model_description.guid,
            unzipDirectory=self.unzipdir,
            modelIdentifier=self.model_description.coSimulation.modelIdentifier,
            instanceName='instance1')
        self.fmu.instantiate()
        self.fmu.setupExperiment(startTime=self.start_time)
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()
        self.status = "Ready"


    def step(self, X, step_size=None):
        self.status = "Busy"

        # TODO Fix!
        vr_input = [self.vrs['Speed'], self.vrs['Torque']]
        vr_output = [self.vrs['Output']]
        data = [X['Speed'], X['Torque']]
        time = X['Time']

        if step_size == None:
            step_size = self.step_size
        # set the input
        self.fmu.setReal(vr_input, data)

        # perform one step
        self.fmu.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)

        # get the values for 'inputs' and 'outputs'
        response = self.fmu.getReal(vr_input + vr_output)
        print("Got response from model", response)
        self.status = "Ready"
        # TODO Fix
        return [response[2]]

    def step_batch(self):
        raise NotImplementedError

    def destroy(self):
        self.fmu.terminate()
        self.fmu.freeInstance()
        # clean up
        shutil.rmtree(self.unzipdir)
