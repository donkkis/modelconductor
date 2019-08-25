__package__ = "modelconductor"
from abc import ABCMeta, abstractmethod
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
from .utils import Measurement
from .exceptions import ExperimentDurationExceededException
from .modelhandler import ModelHandler

# --- Static variables ---
TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
# TIME_FORMAT = "%d.%m.%Y %H:%M:%S"


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
        self.global_idx = 0
        self.validation_strategy = "last_datapoint"
        self.last_measurement = None

    def add_consumer(self, consumer):
        """
        Args:
            consumer (ModelHandler): a ModelHandler instance who is to consume the data
            in current buffer
        """
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
        if "index" not in measurement.keys():
            measurement["index"] = self.global_idx
        self.buffer.put_nowait(measurement)
        self.global_idx = self.global_idx + 1

    def receive_batch(self, measurements):
        """

        Args:
            measurements (list[Measurement]):  A list of measurement datapoints
        """
        for measurement in measurements:
            if "index" not in measurement.keys():
                measurement["index"] = self.global_idx
                self.global_idx = self.global_idx + 1

        if self.validation_strategy == "last_datapoint":
            for measurement in measurements:
                if self.last_measurement is None:
                    self.last_measurement = measurement
                for k, v in measurement.items():
                    if v is None:
                        try:
                            measurement[k] = self.last_measurement[k]
                        except Exception:
                            raise Exception("Nonevalue encountered but no last datapoint was available")

        [self.buffer.put_nowait(measurement) for measurement in measurements]

    @abc.abstractmethod
    def receive(self): pass

    def give(self):
        """
        Get a single measurement element from the FIFO buffer
        """
        try:
            measurement = self.buffer.get_nowait()
            self.last_measurement = measurement
            return measurement
        except Empty:
            return None

    def give_batch(self, batch_size):
        """
        Get a list of batch_size first measurement elements in the FIFO buffer
        """

        measurements = [self.give() for i in range(batch_size)]
        return measurements


class IncomingMeasurementListener(MeasurementStreamHandler):
    """Should distribute the incoming
    signals to the relevant simulation models
    """
    pass

class MeasurementStreamPoller(MeasurementStreamHandler):

    @abc.abstractmethod
    def poll(self): pass
    @abc.abstractmethod
    def poll_batch(self): pass


class IncomingMeasurementBatchPoller(MeasurementStreamPoller):


    def __init__(self, db_uri, query_path, polling_interval=90, polling_window=60, start_time=None,
                 stop_time=None, query_cols="*", buffer=None, consumers=None):
        super().__init__(buffer, consumers)
        self.db_uri = db_uri
        self.polling_interval = polling_interval
        # TODO should inherit this from Experiment
        self.start_time = dt.now() if start_time is None else start_time
        # TODO should inherit this from Experiment
        self.stop_time = self.start_time + timedelta(minutes=120) if stop_time is None else stop_time
        # TODO is this even needed?
        self.query_cols = query_cols
        self.polling_window = polling_window
        self._stopevent = None
        self.engine = None
        self.conn = None
        self.polling_start_timestamp = self.start_time.strftime(TIME_FORMAT)
        self.polling_stop_timestamp = \
            (self.start_time + timedelta(seconds=self.polling_window)).strftime(TIME_FORMAT)
        with open(query_path, 'rb') as f:
            self.query = f.read().decode().replace("\r\n", " ")

    def receive(self):
        self.poll()

    def poll(self):
        self.engine = sqlalchemy.create_engine(self.db_uri)
        self.conn = self.engine.connect()
        assert(self.conn.closed is False)

        # debug
        print("Polling database connection at " + str(self.conn) + " at " + str(
             self.polling_interval) + " s interval, CTRL+C to stop")
        try:
            while not isinstance(self._stopevent, type(threading.Event())):
                try:
                    print("Excecuting query...")
                    self.poll_batch()
                    self.update_timestamps()
                    print("Done")
                except IndexError:
                    print("No more records available, sleeping...")
                    sleep(0.5)
                except sqlite3.OperationalError:
                    print("Waiting for database to become operational...")
                    sleep(5)
                sleep(self.polling_interval)
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
        res = self.conn.execute(q) # sqlalchemy.engine.ResultProxy
        data = res.fetchall()
        data = [dict(zip(tuple(res.keys()), datum)) for datum in data]
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
        # print("Excecuting:", self.query)
        result = self.c.execute(self.query)
        query_keys = [col[0] for col in result.description]
        result = Measurement(dict(zip(query_keys, result.fetchall()[0])))
        self.first_unseen_pk += 1
        self.query = "SELECT " + self.query_cols + " FROM data WHERE `index`=" + str(self.first_unseen_pk)
        self.receive_single(result)
