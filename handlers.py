from typing import List, Any
from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
import shutil
from queue import Queue
from time import sleep
import sqlite3
import pickle
import random
import abc
import threading
import numpy as np

class Event(list):
    """Event subscription.

    A list of callable objects. Calling an instance of this will cause a
    call to each item in the list in ascending order by index.

    """
    def __call__(self, *args, **kwargs):
        for f in self:
            f(*args, **kwargs)

    def __repr__(self):
        return "Event(%s)" % list.__repr__(self)


class MeasurementConfiguration:
    """
    TODO: Make a configuration file that is read on server initiation
    """

    def __init__(self):
        pass


class Experiment:
    """

    """

    def __init__(self, runtime=9999, routes=[]):
        """

        Args:
            runtime:
            routes (List(tuple(MeasurementStreamHandler, ModelHandler)):
        """

        self.max_runtime = runtime
        self.routes = routes
        self.results = []

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
        self.routes.append(route)


class OnlineSingleExperiment(Experiment):
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

        # Whenever new data is received, feed-forward to model
        while True:
            if not src.buffer.empty() and mdl.status == "Ready":
                # simulation step

                 data = mdl.pull()
                 # debug
                 # print(data)

                 # TODO This fails
                 res = mdl.step(data[0])
                 self.results.append(res)
            else:
                pass


class ModelHandler:

    def __init__(self, sources=[]):
        """
        Args:
            sources (List[MeasurementStreamHandler]): A list of MeasurementStreamHandler objects
            associated with this ModelHandler instance
        """
        self.sources = sources
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
        Request the newest available datapoint from all sources
        TODO: should be able to handle subsets of sources
        Returns: List[object]

        """
        # TODO Might return None
        res = [datapoint.give() for datapoint in self.sources]
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


class MeasurementStreamHandler:

    def __init__(self, buffer=Queue(), consumers=[]):
        """

        Args:
            consumers (List[ModelHandler]): A list of ModelHandler objects associated with this
            MeasurementStreamHandler instance
            buffer (Queue) : A FIFO queue of measurement objects
        """
        self.buffer = buffer
        self.consumers = consumers
        # self.data_source_uri = data_source_uri
        # self.data_source_is_active = data_source_is_active

    def add_consumer(self, consumer):
        self.consumers.append(consumer)
        return consumer

    def remove_consumer(self, consumer):
        self.consumers.remove(consumer)
        return consumer

    def receive_single(self, measurement):
        """
        Args:
            measurement (dict(key, value)): A single datapoint
        """
        self.buffer.put_nowait(measurement)

    def give(self):
        try:
            return self.buffer.get_nowait()
        except Queue.Empty:
            return None


class IncomingMeasurementListener(MeasurementStreamHandler):
    """Should distribute the incoming
    signals to the relevant simulation models
    """
    pass


class IncomingMeasurementPoller(MeasurementStreamHandler):
    def __init__(self, polling_interval, db_uri, query_cols="*", buffer=Queue(), consumers=[]):
        super().__init__(buffer, consumers)
        self.db_uri = db_uri
        self.polling_interval = polling_interval
        self.conn = sqlite3.connect(db_uri)
        self.first_unseen_pk = 0
        self.c = self.conn.cursor()
        self.query_cols = query_cols
        self.query = "SELECT " + self.query_cols + " FROM data WHERE `index`=" + str(self.first_unseen_pk)

    def handle_single(self, measurement):
        self.buffer.put(measurement)
        print(self.buffer.qsize())

    def poll(self):
        self.conn = sqlite3.connect(self.db_uri)
        self.c = self.conn.cursor()

        print("Polling database connection at " + str(self.conn) + " at " + str(
            self.polling_interval) + " s interval, CTRL+C to stop")
        try:
            while True:
                sleep(self.polling_interval)
                self.poll_newest_unseen()
        except KeyboardInterrupt:
            return

    def poll_newest_unseen(self):
        # avoid threading errors
        result = self.c.execute(self.query)
        query_keys = [col[0] for col in result.description]
        result = dict(zip(query_keys, result.fetchall()[0]))
        self.first_unseen_pk += 1
        self.handle_single(result)


class SklearnModelHandler(ModelHandler):

    def __init__(self, model_filename, input_keys=None, target_keys=None, sources=[]):
        super().__init__(sources)
        with open(model_filename, 'rb') as pickle_file:
            self.model = pickle.load(pickle_file)
        self.input_keys = input_keys
        self.target_keys = target_keys

    def step(self, X):
        # X is dict when calling from run
        self.status = "Busy"
        # convert to numpy and sort columns to same order as input_keys
        # to make sure input is in format that the model expects
        X = np.array([X[k] for k in self.input_keys], ndmin=2)
        result = list(self.model.predict(X))
        self.status = "Ready"
        print(result)
        return result

    def spawn(self):
        self.status = "Ready"


    def destroy(self):
        pass


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
