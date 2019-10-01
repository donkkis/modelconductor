__package__ = "modelconductor"
import abc
import os
import threading
import uuid
from pandas import DataFrame
from datetime import datetime as dt
from datetime import timedelta
from warnings import warn
from .measurementhandler import MeasurementStreamHandler
from .modelhandler import ModelHandler
from .modelhandler import ModelStatus
from functools import wraps
from .utils import ModelResponse

class Experiment:
    """
    A base class for Experiments
    """

    def __init__(self,
                 routes=None,
                 runtime=None,
                 logging=False,
                 log_path=None):
        """
        Args:
            routes: List of (MeasurementStreamHandler, ModelHandler)
                tuples, defining one-to-one mappings from data sources
                to destinations
            runtime: Integer time in minutes after which the experiment
                is terminated and all associated threads are killed. If
                None, the experiment will run indefinitely
            logging: Boolean, if True experiment results are output to a
                file
            log_path: The file_path string where to output the results.
                Has no effect is logging is False
        """
        self.stop_time = None
        self.runtime = runtime
        self.routes = routes if routes is not None else []
        self.results = []
        self.log_path = log_path
        self.logging = logging
        self.logger = None
        self.has_been_setup = False

    def __str__(self):
        return str(type(self))

    @abc.abstractmethod
    def run(self): pass

    def _run(run):
        """Wrapper method for run implementations

        Contains boilerplate operations needed to execute experiments
        """
        @wraps(run)
        def wrapper(inst, *args, **kwargs):
            # inst is the concrete object calling its run method
            if inst.runtime is not None:
                inst.stop_time = dt.now() + timedelta(minutes=inst.runtime)
            else:
                inst.stop_time = None
            # Run setup if not previously done
            if not inst.has_been_setup:
                inst.setup()
            # Initiate logging for each route, if applicable
            if inst.logging:
                # MeasurementHandler, ModelHandler
                for src, dest in inst.routes:
                    headers = ["timestamp"]
                    if dest.input_keys:
                        headers += dest.input_keys
                    if dest.target_keys:
                        headers += dest.target_keys
                    if dest.control_keys:
                        headers += dest.control_keys
                    inst.logger = inst.initiate_logging(headers=headers,
                                                        path=inst.log_path)
            return run(inst, *args, **kwargs)
        return wrapper

    def initiate_logging(self, path=None, headers=None):
        """Instantiate the log file and write headers

        Args:
            path: Path string of where the outfile will be written
            headers: List of strings, headers for the generated csv file

        Returns:
            The file handle to the instantiated logfile
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
            f.flush()
        return f

    def terminate_logging(self, file=None):
        """Stop logging and finalize the file handle object

        Args:
            file: File handle object instance to be finalized
        Returns:
            The finalized file handle
        """
        if file is None:
            file = self.logger
        file.close()
        return file

    def log_row(self, row, model=None):
        """Write a single row to log file

        Args:
            row: List or dict containing the headers as keys of strings
                that will be output to new row in file as csv
            model: The associated ModelHandler instance
        """
        if isinstance(row, (ModelResponse, dict)):
            ordered_keys =\
                model.input_keys + model.target_keys + model.control_keys
            row = DataFrame(row, index=[1])
            row = list(row[ordered_keys].loc[1])
            # append timestamp
            row = list(str(dt.now())) + row
        try:
            # ensure inputs are strings before writing
            row = list(map(str, row))
            print(",".join(row), file=self.logger)
            self.logger.flush()
        except Exception:
            warn("Log file could not be written", ResourceWarning)

    def setup(self):
        """Setup the routes

        Go through each tuple in routes and add source to consumer and
        consumer to source

        # TODO This can probably be called directly from run and not by user
        """
        if self.has_been_setup:
            warn("Repeated call to setup detected."
                 "This might lead to unexpected behavior")
            status = 1
        else:
            status = 0

        for route in self.routes:
            source = route[0]  # type: MeasurementStreamHandler
            consumer = route[1]  # type: ModelHandler

            if not isinstance(consumer, ModelHandler):
                t = str(type(consumer))
                raise TypeError("Expected a ModelHandler type, "
                                "got {}".format(t))
            source.add_consumer(consumer)

            if not isinstance(source, MeasurementStreamHandler):
                t = str(type(source))
                raise TypeError("Expected a MeasurementStreamHandler type, "
                                "got {} instead".format(t))
            consumer.add_source(source)
        self.has_been_setup = True
        return status

    def add_route(self, route):
        """Add a data route to the current experiment

        Args:
            route: A (MeasurementStreamHandler, ModelHandler) object
        """
        if not isinstance(route[0], MeasurementStreamHandler):
            raise TypeError("Route position 0 is not a valid "
                            "MeasurementStreamHandler object")
        if not isinstance(route[1], ModelHandler):
            raise TypeError("Route position 1 is not a valid "
                            "ModelHandler object")
        self.routes.append(route)


class OnlineOneToOneExperiment(Experiment):
    """
    Online single-source, single model experiment
    """

    def _stopping_condition(self, event=None):
        """Determine if main loop will proceed to next iteration"""
        if self.stop_time:
            return dt.now() >= self.stop_time
        elif isinstance(self._stopevent, type(threading.Event())):
            return True
        return False

    def _model_loop(self, src, mdl):
        """Feed data to model as it becomes available in buffer

        Args:
            src: A MeasurementStreamHandler instance
            mdl: A ModelHandler instance
        """
        while not self._stopping_condition():
            if not src.buffer.empty() and mdl.status == ModelStatus.READY:
                # simulation step
                data = mdl.pull()
                try:
                    assert (data is not None)
                except AssertionError:
                    warn("ModelHandler.pull called on empty buffer",
                         UserWarning)
                    continue
                res = mdl.step(data[0])
                # print("Got response from model: ", res)  # debug
                self.results.append(res)

                if self.logging:
                    # TODO need to generalize the measurement timestamp
                    # TODO will fail with more than one control key!
                    # TODO Write integration test
                    self.log_row(res)

    @Experiment._run
    def run(self):
        """Run the experiment"""
        try:
            assert(len(self.routes) == 1)
        except AssertionError:
            raise AssertionError("Multi-route experiment not allowed")

        # Initiate model
        mdl = self.routes[0][1]  # type: ModelHandler
        mdl.spawn()

        # Start receiving
        # By setting daemon=True we ensure that receive_thread is killed
        # when main loop exits
        src = self.routes[0][0]  # type: MeasurementStreamHandler
        receive_loop = threading.Thread(target=src.receive, daemon=True)
        receive_loop.start()

        # start the main loop
        self._model_loop(src, mdl)

        src._stopevent = threading.Event()
        mdl.destroy()
        print("Process exited due to experiment time out")
        return True