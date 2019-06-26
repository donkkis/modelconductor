from typing import List
from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
import shutil
from queue import Queue
from queue import Empty
from time import sleep
from datetime import datetime as dt
from datetime import timedelta
import sqlite3
import pickle
import abc
import threading
import numpy as np
import sqlalchemy
from keras import Sequential
from warnings import warn
import os.path
import uuid

class MeasurementConfiguration:
    """
    TODO: Make a configuration file that is read on server initiation
    """

    def __init__(self):
        raise NotImplementedError


class Experiment:
    """

    """

    def __init__(self, start_time=None,
                 routes=None,
                 runtime=10,
                 logging=False,
                 log_path=None):
        """

        Args:
            stop_time:
            routes (List(tuple(MeasurementStreamHandler, ModelHandler)):
        """

        self.start_time = start_time if start_time is not None else dt.now()
        self.stop_time = self.start_time + timedelta(minutes=runtime)
        self.routes = routes if routes is not None else []
        self.results = []
        self.log_path = log_path
        self.logging = logging
        self.logger = None

    def __str__(self):
        return str(type(self))

    def initiate_logging(self, path=None, headers=None):
        """

        Args:
            path (String): Path where the outfile will be written
            headers (List[String]): Headers for the generated csv file

        Returns:

        """
        tic = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        if path is None:
            self.log_path = "experiment_{}.log".format(tic)
        else:
            self.log_path = path
        if os.path.isfile(self.log_path):
            self.log_path = (str(uuid.uuid1()))[0:9] + self.log_path
        f = open(self.log_path, 'w+')
        self.logger = f
        if headers:
            print(",".join(headers), file=f)
        return f

    def terminate_logging(self, file=None):
        """

        Args:
            file (file object): File object instance to be finalized

        Returns:

        """
        if file is None:
            file = self.logger
        file.close()
        return file

    def log_row(self, row):
        """

        Args:
            row (List(String)):

        Returns:

        """
        try:
            print(",".join(row), file=self.logger)
            self.logger.flush()
        except Exception:
            warn("Log file could not be written", ResourceWarning)

    @abc.abstractmethod
    def run(self):
        pass

    def setup(self):
        for route in self.routes:
            src = route[0]  # type: MeasurementStreamHandler
            mdl = route[1]  # type: ModelHandler
            src.add_consumer(mdl)
            mdl.add_source(src)

    def add_route(self, route):
        if not isinstance(route[0], MeasurementStreamHandler) or not isinstance(route[1], ModelHandler):
            raise TypeError
        self.routes.append(route)


class OnlineOneToOneExperiment(Experiment):
    """
    Online single-source, single model experiment
    """


    def run(self):
        assert(len(self.routes) == 1)
        # Initiate model
        mdl = self.routes[0][1]  # type: ModelHandler
        mdl.spawn()

        # Start polling
        src = self.routes[0][0]  # type: MeasurementStreamHandler
        threading.Thread(target=src.poll).start()

        # Initiate logging if applicable
        if self.logging:
            # TODO should move most of this stuff to initiate_logging?
            assert(len(mdl.target_keys) == 1)
            gt_title = "{}_meas".format(mdl.target_keys[0])
            pred_title = "{}_pred".format(mdl.target_keys[0])
            self.logger = self.initiate_logging(headers=["timestamp", gt_title, pred_title], path=self.log_path)

        # Whenever new data is received, feed-forward to model
        print("now ", dt.now())  # debug
        while dt.now() < self.stop_time:
            if not src.buffer.empty() and mdl.status == "Ready":
                # simulation step
                data = mdl.pull()
                try:
                    assert(data is not None)
                except AssertionError:
                    warn("ModelHandler.pull called on empty buffer", UserWarning)
                    continue
                # debug
                # print(data)

                res = mdl.step(data[0])
                # print(res)  # debug
                self.results.append(res)

                if self.logging:
                    # TODO need to generalize the measurement timestamp
                    timestamp_key = "Time"

                    row = [str(data[0][timestamp_key]),
                           str(data[0][mdl.target_keys[0]]),
                           str(res[0])]

                    # print(row) #  debug
                    self.log_row(row)
            else:
                continue

        src._stopevent = threading.Event()
        print("Process exited due to experiment time out")
        return True


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
        if not isinstance(source, MeasurementStreamHandler):
            t = str(type(source))
            raise TypeError("Expected a MeasurementStreamHandler type, got {} instead".format(t))
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

    @abc.abstractmethod
    def step(self):
        """Feed-forward the associated model with the latest datapoint and return the response"""

    @abc.abstractmethod
    def spawn(self):
        """Instantiate the associated model so that steps can be executed"""

    @abc.abstractmethod
    def destroy(self):
        """Remove the model instance from the current configuration and delete self"""


class Measurement(dict):
    pass

    def to_numpy(self, keys=None):
        """

        Args:
            keys: Ordered list, a subset of the Measurement object's dict keys
            to be taken in to account when creating the numpy array

        Returns:
            numpy_meas: A subset of the measurement as a numpy array

        """
        if keys is None:
            keys = self.keys()
        numpy_meas = np.array([self[k] for k in keys], ndmin=2)
        return numpy_meas


class TrainableModelHandler(ModelHandler):

    @abc.abstractmethod
    def fit(self):
        """Fit the model's trainable parameters"""


class MeasurementStreamHandler:

    def __init__(self, buffer=None, consumers=None):
        """

        Args:
            consumers (List[ModelHandler]): A list of ModelHandler objects associated with this
            MeasurementStreamHandler instance
            buffer (Queue) : A FIFO queue of measurement objects
        """

        # https://stackoverflow.com/questions/13525842/variable-scope-in-python-unittest
        # DO NOT use mutable types as default arguments
        self.buffer = buffer if buffer is not None else Queue()
        self.consumers = consumers if consumers is not None else []

    def add_consumer(self, consumer):
        """
        Args:
            consumer (ModelHandler): a ModelHandler instance who is to consume the data
            in current buffer
        """
        if not isinstance(consumer, ModelHandler):
            t = str(type(consumer))
            raise TypeError("Expected a ModelHandler type, got {}".format(t))
        self.consumers.append(consumer)
        return consumer

    def remove_consumer(self, consumer):
        self.consumers.remove(consumer)
        return consumer

    def receive_single(self, measurement):
        """
        Args:
            measurement (Measurement): A single datapoint dict
        """
        self.buffer.put_nowait(measurement)

    def receive_batch(self, measurements):
        """

        Args:
            measurements (list[Measurement]):  A list of measurement datapoints
        """
        [self.buffer.put_nowait(measurement) for measurement in measurements]

    def give(self):
        try:
            return self.buffer.get_nowait()
        except Empty:
            return None


class IncomingMeasurementListener(MeasurementStreamHandler):
    """Should distribute the incoming
    signals to the relevant simulation models
    """
    pass


class MeasurementStreamPoller(MeasurementStreamHandler):

    @abc.abstractmethod
    def poll(self):
        pass

    @abc.abstractmethod
    def poll_batch(self):
        pass


class ExperimentDurationExceededException(Exception):
    pass


class IncomingMeasurementBatchPoller(MeasurementStreamPoller):
    def __init__(self, db_uri, query_path, polling_interval=90, polling_window=60, start_time=None,
                 stop_time=None, query_cols="*", buffer=None, consumers=None):
        super().__init__(buffer, consumers)
        self.db_uri = db_uri
        self.polling_interval = polling_interval
        self.start_time = dt.now() if start_time is None else start_time
        self.stop_time = self.start_time + timedelta(minutes=10) if stop_time is None else stop_time
        self.query_cols = query_cols
        self.polling_window = polling_window
        self._stopevent = None
        self.engine = None
        self.conn = None
        self.polling_start_timestamp = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        self.polling_stop_timestamp = \
            (self.start_time + timedelta(seconds=self.polling_window)).strftime("%Y-%m-%d %H:%M:%S")
        with open(query_path, 'rb') as f:
            self.query = f.read().decode().replace("\r\n", " ")

    def poll(self):
        self.engine = sqlalchemy.create_engine(self.db_uri)
        self.conn = self.engine.connect()
        assert(self.conn.closed is False)

        # debug
        # print("Polling database connection at " + str(self.conn) + " at " + str(
        #     self.polling_interval) + " s interval, CTRL+C to stop")
        try:
            while not isinstance(self._stopevent, type(threading.Event())):
                sleep(self.polling_interval)
                try:
                    self.poll_batch()
                except IndexError:
                    print("No more records available, sleeping...")
                    sleep(0.5)
                except sqlite3.OperationalError:
                    print("Waiting for database to become operational...")
                    sleep(5)
        except Exception:
            raise Exception("Unknown error")
        print("Polling thread excited due to stopevent")
        return True

    def poll_batch(self):
        # avoid threading errors
        q = self.query.format(self.polling_start_timestamp, self.polling_stop_timestamp)
        res = self.conn.execute(q) # sqlalchemy.engine.ResultProxy
        data = res.fetchall()
        data = [dict(zip(tuple(res.keys()), datum)) for datum in data]
        self.receive_batch(data)
        return data

    def update_timestamps(self, old_start_timestamp=None):
        """
        Update next batch polling end/stop timestamps according to polling window
        Args:
            old_start_timestamp: Beginning timestamp of the previous batch
        Returns:
            self.polling_start_timestamp: Updated timestamp where the next batch should begin
            self.polling_stop_timestamp: Updated timestamp where the next batch should end
        """

        form = "%Y-%m-%d %H:%M:%S"
        if old_start_timestamp is None:
            old_start_timestamp = self.polling_start_timestamp
        next_start = \
            (dt.strptime(old_start_timestamp, form) + timedelta(seconds=self.polling_window))
        next_end = \
            (dt.strptime(old_start_timestamp, form) + timedelta(seconds=2*self.polling_window))
        if next_end > self.stop_time:
            next_end = self.stop_time
        if next_start >= self.stop_time:
            raise ExperimentDurationExceededException("Updated start timestamp exceeds experiment duration")

        self.polling_start_timestamp = next_start.strftime(form)
        self.polling_stop_timestamp = next_end.strftime(form)
        return self.polling_start_timestamp, self.polling_stop_timestamp


class IncomingMeasurementPoller(MeasurementStreamPoller):
    def __init__(self, polling_interval, db_uri, first_unseen_pk=0, query_cols="*", buffer=None, consumers=None):
        super().__init__(buffer, consumers)
        self.db_uri = db_uri
        self.polling_interval = polling_interval
        self.conn = sqlite3.connect(db_uri)
        self.first_unseen_pk = first_unseen_pk
        self.c = self.conn.cursor()
        self.query_cols = query_cols
        self.query = "SELECT " + self.query_cols + " FROM data WHERE `index`=" + str(self.first_unseen_pk)
        self._stopevent = None

    def poll(self):
        self.conn = sqlite3.connect(self.db_uri)
        self.c = self.conn.cursor()

        print("Polling database connection at " + str(self.conn) + " at " + str(
            self.polling_interval) + " s interval, CTRL+C to stop")
        try:
            while not isinstance(self._stopevent, type(threading.Event())):
                sleep(self.polling_interval)
                try:
                    self.poll_batch()
                except IndexError:
                    print("No more records available, sleeping...")
                    sleep(0.5)
                except sqlite3.OperationalError:
                    print("Waiting for database to become operational...")
                    sleep(5)
        except Exception:
            raise Exception("Unknown error")
        print("Polling thread excited due to stopevent")
        return True

    def poll_batch(self):
        # avoid threading errors
        result = self.c.execute(self.query)
        query_keys = [col[0] for col in result.description]
        result = Measurement(dict(zip(query_keys, result.fetchall()[0])))
        self.first_unseen_pk += 1
        self.query = "SELECT " + self.query_cols + " FROM data WHERE `index`=" + str(self.first_unseen_pk)
        self.receive_single(result)


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

    def step(self, X):
        # X is dict when calling from run
        self.status = "Busy"
        # convert to numpy and sort columns to same order as input_keys
        # to make sure input is in format that the model expects
        X = X.to_numpy(self.input_keys)
        result = list(self.model.predict(X))
        self.status = "Ready"
        # print(result)
        return result

    def fit(self, X):
        # TODO need something like partial_fit from scickit-multiflow
        pass

    def spawn(self):
        self.status = "Ready"


    def destroy(self):
        pass


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
        X = X.to_numpy(self.input_keys)
        result = list(self.model.predict(X))
        self.status = "Ready"
        print(result)
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
        yield self.model
        self.status = "Ready"

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
        X, y = self.measurements_to_numpy(measurements)
        self.model.train_on_batch(X, y)
        yield self.model
        self.status = "Ready"





class FMUModelHandler(ModelHandler):
    """Loads and executes FMU binary modules
    """

    def __init__(self, fmu_filename, start_time, threshold, stop_time, step_size):
        super(FMUModelHandler, self).__init__()
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


    def step(self, vr_input, vr_output, data, time, step_size=None):
        if step_size == None:
            step_size = self.step_size
        # set the input
        self.fmu.setReal(vr_input, data)

        # perform one step
        self.fmu.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)

        # get the values for 'inputs' and 'outputs'
        response = self.fmu.getReal(vr_input + vr_output)
        print(response)
        return response

    def destroy(self):
        self.fmu.terminate()
        self.fmu.freeInstance()
        # clean up
        shutil.rmtree(self.unzipdir)


def run():
    print("Ready")
    while True:
        """
        Placeholder stuff for manual testing. Could be broken.
        """
        step = input()
        step = step.split(maxsplit=1)
        if step[0] == "handle":
            # meas_hand.receive_single(step[1])
            pass
        elif step[0] == "spawn":
            # fmu_hand.spawn()
            pass
        elif step[0] == "step":
            # Fetch a single datapoint from buffer
            # datapoint = meas_hand.buffer.get()
            # Feed forward to simulator
            # fmu_hand.step([0,1], [2], [random.random()*2200, random.random()*500], 1)
            pass
        elif step[0] == "poll":
            # meas_poller.poll()
            pass
        elif step[0] == "step_nox":
            # sk_hand.step(X)
            pass
        elif step[0] == "step_iter_nox":
            pass


if __name__ == "__main__":
    run()
