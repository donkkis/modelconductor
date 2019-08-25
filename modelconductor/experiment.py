__package__ = "modelconductor"
import abc
import os
import threading
import uuid
from datetime import datetime as dt
from datetime import timedelta
from warnings import warn
from .measurementhandler import MeasurementStreamHandler
from .modelhandler import ModelHandler


class Experiment:
    """
    A base class for Experiments
    """

    def __init__(self, start_time=None,
                 routes=None,
                 runtime=10,
                 logging=False,
                 log_path=None):
        """
        Args:
            routes (List(tuple(MeasurementStreamHandler, ModelHandler)): Mappings between data sources and data
            sinks
            runtime (int): Time in minutes after which the experiment is terminated and all associated
            threads are terminated as well
            logging (Boolean): If the experiment results should be output to a file
            log_path (str): The filepath where to output the results. Has no effect is logging is False
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
            source = route[0]  # type: MeasurementStreamHandler
            consumer = route[1]  # type: ModelHandler

            if not isinstance(consumer, ModelHandler):
                t = str(type(consumer))
                raise TypeError("Expected a ModelHandler type, got {}".format(t))
            source.add_consumer(consumer)

            if not isinstance(source, MeasurementStreamHandler):
                t = str(type(source))
                raise TypeError("Expected a MeasurementStreamHandler type, got {} instead".format(t))
            consumer.add_source(source)

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
            # assert(len(mdl.target_keys) == 1)
            # gt_title = "{}_meas".format(mdl.target_keys[0])
            # pred_title = "{}_pred".format(mdl.target_keys[0])
            headers = ["timestamp"] + mdl.input_keys + mdl.target_keys + mdl.control_keys
            print(headers)
            self.logger = self.initiate_logging(headers=headers, path=self.log_path)

        # Whenever new data is received, feed-forward to model
        # print("now ", dt.now())  # debug
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
                # print("Got response from model: ", res)  # debug
                self.results.append(res)

                if self.logging:
                    # TODO need to generalize the measurement timestamp
                    # TODO will fail with more than one control key!
                    row = [str(dt.now())] + [str(item) for item in res[0]] + [str(data[0][mdl.control_keys[0]])]

                    # print(row) #  debug
                    self.log_row(row)
            else:
                continue

        src._stopevent = threading.Event()
        mdl.destroy()
        print("Process exited due to experiment time out")
        return True

class OnlineBatchTrainableExperiment(Experiment):

    def __init__(self, start_time=None,
                 routes=None,
                 runtime=10,
                 logging=False,
                 log_path=None,
                 batch_size=10,
                 timestamp_key=None):

        super().__init__(start_time,
                         routes,
                         runtime,
                         logging)
        self.batch_size = batch_size
        self.log_path = log_path
        self.timestamp_key = timestamp_key

    def log_batch(self, data, groundtruth_key, timestamp_key, results, debug=False):
        """

        Args:
            data (List[Measurement]):
            result (List):
            groundtruth_key (str):
            timestamp_key (str):

        Returns:

        """
        for datum, result in zip(data, results):
            row = [str(datum[timestamp_key]),
                   str(datum[groundtruth_key]),
                   str(result)]
            if debug:
                print(row)
            self.log_row(row)

    def run(self):
        assert(len(self.routes) == 1)
        # Initiate model
        mdl = self.routes[0][1]  # type: TrainableModelHandler
        mdl.spawn()

        # Start polling
        src = self.routes[0][0]  # type: IncomingMeasurementBatchPoller
        threading.Thread(target=src.poll).start()

        # Initiate logging if applicable
        if self.logging:
            # TODO should move most of this stuff to initiate_logging?
            # assert(len(mdl.target_keys) == 1)
            gt_title = "{}_meas".format(mdl.target_keys[0])
            pred_title = "{}_pred".format(mdl.target_keys[0])
            self.logger = self.initiate_logging(headers=["timestamp", gt_title, pred_title], path=self.log_path)

        # Whenever new data is received, feed-forward to model
        while dt.now() < self.stop_time:
            # print(src.buffer.qsize())  #  debug
            # Check that new batch is available and the model is ready
            if src.buffer.qsize() >= self.batch_size and mdl.status == "Ready":
                # simulation step
                data = mdl.pull_batch(self.batch_size)  # List[List[Measurement]]
                data = data[0]  # List[Measurement]
                try:
                    assert(data is not None)
                except AssertionError:
                    warn("ModelHandler.pull called on empty buffer", UserWarning)
                    continue
                # debug
                # print(data)

                _, res = mdl.step_fit_batch(data)  # _, List
                # print(res)  # debug
                self.results += res

                if self.logging:
                    # TODO implement and test batch logging
                    self.log_batch(data=data,
                                   groundtruth_key=mdl.target_keys[0],
                                   timestamp_key=self.timestamp_key,
                                   results=res,
                                   debug=True)
            else:
                continue

        src._stopevent = threading.Event()
        print("Process exited due to experiment time out")
        return True
