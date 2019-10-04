__package__ = "modelconductor"
from queue import Queue
from queue import Empty
from time import sleep
from datetime import datetime as dt
from datetime import timedelta
from .utils import Measurement
from .exceptions import ExperimentDurationExceededException
from modelconductor import server
from threading import Thread
from enum import Enum
import queue
import sqlite3
import abc
import threading
import sqlalchemy
import json
import os

# --- Static variables ---
TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
# TIME_FORMAT = "%d.%m.%Y %H:%M:%S"
PORT = 8080

class ValidationStrategy(Enum):
    # Potentially dangerous
    NO_VALIDATION = 1
    # If invalid value is encountered, re-uses the value from last
    # available valid datapoint
    LAST_DATAPOINT = 2

VAL_STRATS = {"NO_VALIDATION": ValidationStrategy.NO_VALIDATION,
              "LAST_DATAPOINT": ValidationStrategy.LAST_DATAPOINT}


class MeasurementStreamHandler:

    def __init__(self, validation_strategy='NO_VALIDATION',
                 buffer=None, consumers=None):
        """Creates a MeasurementStreamHandler.

        Args:
            validation_strategy: Action taken when an invalid value is
                encountered. Must be one of the following strings:
                'NO_VALIDATION' - Do nothing. Might break things!
                'LAST_DATAPOINT' - Re-use the latest valid value
            consumers: (Optional) A list of ModelHandler objects linked
                with this MeasurementStreamHandler instance. If set to
                None, consumers may be added later by calling setup().
            buffer: (Optional) a Queue of measurement objects. Should
                typically be set to None for other than testing purposes,
                which results in an empty buffer being created in __init__.
        """

        # https://stackoverflow.com/questions/13525842/variable-scope-in-python-unittest
        # DO NOT use mutable types as default arguments
        self.buffer = buffer if buffer is not None else Queue()
        self.consumers = consumers if consumers is not None else []
        self.validation_strategy = VAL_STRATS[validation_strategy]
        self.last_measurement = None
        self._stopevent = None

    @abc.abstractmethod
    def receive(self): pass

    def add_consumer(self, consumer):
        """Adds a ModelHandler instance that will consume data in buffer.

        Args:
            consumer: The ModelHandler instance to be added.

        Returns:
            The consumer that was added.
        """
        self.consumers.append(consumer)
        return consumer

    def validate_measurement(self, measurement):
        """Validate measurement before appending to buffer

        Basic implementation of the validation rules imposed by the
        chosen ValidationStrategy attribute. Subclasses could choose to
        override these to better fit their specifc needs for data
        validation.

        Args:
            measurement: The Measurement instance to be validated

        Returns:
            measurement: The validated Measurement
        """
        validation_done = False
        if self.validation_strategy == ValidationStrategy.NO_VALIDATION:
            # Do nothing literally
            pass

        elif self.validation_strategy == ValidationStrategy.LAST_DATAPOINT:
            # Scan for Nones and if found, replace with last valid value
            for k, v in measurement.items():
                if v is None:
                    try:
                        measurement[k] = self.last_measurement[k]
                    except Exception:
                        raise Warning("No last meas available for replacement")

        # the validated measurement becomes new last_measurement
        self.last_measurement = measurement
        return measurement

    def remove_consumer(self, consumer):
        """Removes a ModelHandler instance from current buffer.

        Args:
            consumer: a ModelHandler instance who is to be removed from
                the current experiment.
        Returns:
            The ModelHandler instance that was just removed.
        """
        self.consumers.remove(consumer)
        return consumer

    def receive_single(self, measurement):
        """Handle a single incoming Measurement object

        Args:
            measurement: A single Measurement instance to be added to
                the buffer
        Returns:
            The Measurement that was just added
        """
        measurement = self.validate_measurement(measurement)
        self.buffer.put_nowait(measurement)
        return measurement

    def receive_batch(self, measurements):
        """Receive a batch of sequential measurements

        Args:
            measurements:  A list of Measurement instances
        Returns:

        """
        # Will be true for the first measurement of the first batch
        if self.last_measurement is None:
            self.last_measurement = measurements[0]

        for measurement in measurements:
            measurement = self.validate_measurement(measurement)
            self.buffer.put_nowait(measurement)

    def give(self):
        """
        Returns:
            measurement: The first Measurement element from buffer. None
                if buffer is empty
        """
        try:
            measurement = self.buffer.get_nowait()
        except Empty:
            measurement = None
        return measurement

    def give_batch(self, batch_size):
        """Get a list of batch_size first measurement elements in the
        FIFO buffer

        Args:
            batch_size: The integer batch size to be retrieved

        Returns:
            measurements: A list of Measurement objects. Will contain
                Nones if batch_size exceeds current buffer
        """
        measurements = [self.give() for _ in range(batch_size)]
        return measurements


class IncomingMeasurementListener(MeasurementStreamHandler):
    """Should distribute the incoming
    signals to the relevant simulation models
    """

    def receive(self):
        self.listen()

    def listen(self):
        """Start the listening service
        Returns:
            Boolean if process exited succesfully
        """

        e = threading.Event()
        q = queue.Queue()

        t = Thread(target=server.run, args=(e, q), daemon=True)
        t.start()

        while not isinstance(self._stopevent, type(threading.Event())):
            try:
                item = q.get(timeout=10)
            except Empty:
                continue
            # expect dict
            data = json.loads(item.decode('utf-8'))
            data = Measurement(data)
            self.receive_single(data)
            print("Current buffer: ", self.buffer.qsize())
        return True


class MeasurementStreamPoller(MeasurementStreamHandler):

    @abc.abstractmethod
    def poll(self):
        raise NotImplementedError("Abstract method not implemented")

    @abc.abstractmethod
    def poll_batch(self):
        raise NotImplementedError("Abstract method not implemented")


class IncomingMeasurementBatchPoller(MeasurementStreamPoller):

    def __init__(self,
                 db_uri,
                 query,
                 polling_interval=90,
                 polling_window=60,
                 start_time=None,
                 stop_time=None,
                 validation_strategy='NO_VALIDATION',
                 buffer=None,
                 consumers=None):
        """Creates an IncomingMeasurementBatchPoller.

        Args:
            db_uri: SQLAlchemy compatible URI string of the format:
                dialect+driver://username:password@host:port/database

                Examples:
                'mysql+pymysql://scott:tiger@localhost/foo'
                'sqlite:///foo.db'

                See https://docs.sqlalchemy.org/en/13/core/engines.html
            query: An SQL query String or path to text file where the
                executed query is located
            polling_interval: The approximate minimum interval at which
                consecutive queries are executed. The actual interval will be
                taken as max of polling_interval and actual query cost
            polling_window: Sliding window size for batch polling in seconds
            start_time: A datetime.datetime object, global starting timestamp
                for polling
            stop_time: A datetime.datetime object, global ending timestamp
                for polling
            validation_strategy: Action taken when an invalid value is
                encountered. Must be one of the following strings:
                'NO_VALIDATION' - Do nothing. Might break things!
                'LAST_DATAPOINT' - Re-use the latest valid value
            consumers: (Optional) A list of ModelHandler objects linked
                with this MeasurementStreamHandler instance. If set to
                None, consumers may be added later by calling setup().
            buffer: (Optional) a Queue of measurement objects. Should
                typically be set to None for other than testing purposes,
                which results in an empty buffer being created in __init__.
        """
        super().__init__(validation_strategy, buffer, consumers)
        self.db_uri = db_uri
        self.polling_interval = polling_interval
        self.start_time = dt.now() if start_time is None else start_time
        self.stop_time = self.start_time + timedelta(minutes=120) if stop_time is None else stop_time
        self.polling_window = polling_window
        self._stopevent = None
        self.engine = None
        self.conn = None
        self.polling_start_timestamp = self.start_time.strftime(TIME_FORMAT)
        self.polling_stop_timestamp = \
            (self.start_time + timedelta(seconds=self.polling_window)).strftime(TIME_FORMAT)
        self._parse_query(query)

    def _parse_query(self, query):
        if os.path.exists(query):
            with open(query, 'rb') as f:
                self.query = f.read().decode().replace("\r\n", " ")
        else:
            self.query = query
        return self.query

    def _connect(self):
        self.engine = sqlalchemy.create_engine(self.db_uri)
        self.conn = self.engine.connect()
        try:
            assert(self.conn.closed is False)
        except AssertionError:
            raise AssertionError("Db connection failed")

    def receive(self):
        return self.poll()

    def poll(self):
        if not self.conn or self.conn.closed:
            self._connect()

        # debug
        print("Polling database connection at " + str(self.conn) + " at " + str(
             self.polling_interval) + " s interval, CTRL+C to stop")
        try:
            while not isinstance(self._stopevent, type(threading.Event())):
                try:
                    # wait for the remaining period in polling_interval
                    # if applicable
                    sleep(self.polling_interval - query_duration)
                except Exception:
                    pass
                try:
                    print("Excecuting query...")
                    tic = dt.now()
                    self.poll_batch()
                    toc = dt.now()
                    # query cost in seconds
                    query_duration = ((toc - tic).seconds * 10**6 +
                                      (toc - tic).microseconds) / 10**6
                    self.update_timestamps()
                    print("Done")
                except IndexError:
                    print("No more records available, sleeping...")
                    query_duration = 0
                except sqlite3.OperationalError:
                    print("Waiting for database to become operational...")
                    query_duration = 0
                    sleep(5)
        except KeyboardInterrupt:
            "Process exited per user request"
        except Exception:
            raise Exception("Unknown error")
        finally:
            self.conn.close()
            print("Polling thread excited due to stopevent")
            return True

    def poll_batch(self):
        # avoid threading errors
        q = self.query.format(self.polling_start_timestamp, self.polling_stop_timestamp)
        print(q[:50])  # debug
        res = self.conn.execute(q)  # sqlalchemy.engine.ResultProxy
        data = res.fetchall()
        data = [Measurement(dict(zip(tuple(res.keys()), d))) for d in data]
        self.receive_batch(data)
        # print(data)  # debug
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

        if old_start_timestamp is None:
            old_start_timestamp = self.polling_start_timestamp
        next_start = \
            (dt.strptime(old_start_timestamp, TIME_FORMAT) + timedelta(seconds=self.polling_window))
        next_end = \
            (dt.strptime(old_start_timestamp, TIME_FORMAT) + timedelta(seconds=2 * self.polling_window))
        if next_end > self.stop_time:
            next_end = self.stop_time
        if next_start >= self.stop_time:
            raise ExperimentDurationExceededException("Updated start timestamp exceeds experiment duration")

        self.polling_start_timestamp = next_start.strftime(TIME_FORMAT)
        self.polling_stop_timestamp = next_end.strftime(TIME_FORMAT)
        return self.polling_start_timestamp, self.polling_stop_timestamp


class IncomingMeasurementPoller(MeasurementStreamPoller):

    def receive(self):
        self.poll()

    def __init__(self,
                 polling_interval,
                 db_uri,
                 first_unseen_pk=0,
                 query_cols="*",
                 validation_strategy='NO_VALIDATION',
                 buffer=None,
                 consumers=None):
        super().__init__(validation_strategy, buffer, consumers)
        self.db_uri = db_uri
        self.polling_interval = polling_interval
        self.conn = sqlite3.connect(db_uri)
        self.first_unseen_pk = first_unseen_pk
        self.c = self.conn.cursor()
        self._parse_query_cols(query_cols)
        self.query = "SELECT " + self.query_cols + " FROM data WHERE `index`=" + str(self.first_unseen_pk)
        self._stopevent = None

    def _parse_query_cols(self, query_cols):
        # No validations here
        if isinstance(query_cols, list):
            self.query_cols = ",".join(query_cols)
        else:
            self.query_cols = query_cols

    def poll(self):
        self.conn = sqlite3.connect(self.db_uri)
        self.c = self.conn.cursor()

        print("Polling database connection at " + str(self.conn) + " at " + str(
            self.polling_interval) + " s interval, CTRL+C to stop")
        try:
            while not isinstance(self._stopevent, type(threading.Event())):
                try:
                    # wait for the remaining period in polling_interval
                    # if applicable
                    sleep(self.polling_interval - query_duration)
                except Exception:
                    pass
                try:
                    tic = dt.now()
                    self.poll_batch()
                    toc = dt.now()
                    # query cost in seconds
                    query_duration = ((toc - tic).seconds * 10**6 +
                                      (toc - tic).microseconds) / 10**6
                except IndexError:
                    # debug
                    # print("No more records available, sleeping...")
                    query_duration = 0
                except sqlite3.OperationalError:
                    print("Waiting for database to become operational...")
                    query_duration = 0
                    sleep(5)
        except Exception:
            raise Exception("Unknown error")
        finally:
            print("Polling thread excited due to stopevent")
            self.conn.close()
            return True

    def poll_batch(self):
        # avoid threading errors
        # print("Excecuting:", self.query)
        result = self.c.execute(self.query)
        query_keys = [col[0] for col in result.description]
        result = Measurement(dict(zip(query_keys, result.fetchall()[0])))
        self.first_unseen_pk += 1
        self.query = "SELECT " + self.query_cols + " FROM data WHERE `index`=" + str(self.first_unseen_pk)
        self.receive_single(result)
