__package__ = "modelconductor"
import pickle
import queue
import unittest
from io import TextIOWrapper
import numpy as np
from .utils import Measurement
from .utils import ModelResponse
from .measurementhandler import IncomingMeasurementListener
from .experiment import Experiment
from .modelhandler import ModelHandler, SklearnModelHandler, FMUModelHandler
from .measurementhandler import IncomingMeasurementPoller
from .experiment import OnlineOneToOneExperiment
from .measurementhandler import MeasurementStreamHandler
from .measurementhandler import IncomingMeasurementBatchPoller
from .exceptions import ExperimentDurationExceededException, ModelStepException
from .modelhandler import ModelStatus
from .test_utils import MockFMU
from .test_utils import MockModelHandler
import sqlalchemy as sqla
import pandas as pd
from time import sleep
import os
from threading import Thread
from threading import Event
from datetime import datetime as dt
import uuid
import shutil
import pandas as pd
import numpy as np
import uuid
from sqlalchemy import create_engine
from random import randint
from fmpy import extract
from fmpy import read_model_description
from fmpy.util import download_test_file
from fmpy import dump as fmpydump
from random import random
from socket import AF_INET, socket, SOCK_STREAM
from .testresources.train import prepare_dataset


class SklearnTests(unittest.TestCase):

    def setUp(self):
        self.out_path = str(uuid.uuid1())
        os.mkdir(self.out_path)
        os.chdir(self.out_path)
        self.model_path = '..\\testresources\\pmsm.pickle'
        self.sample_path = '..\\testresources\\pmsm_temperature_sample.csv'
        self.data_path = '..\\testresources\\pmsm_temperature_data.csv'
        conds = [os.path.exists(self.model_path),
                 os.path.exists(self.sample_path),
                 os.path.exists(self.data_path)]
        if not all(conds):
            prepare_dataset()

        data = pd.read_csv(self.sample_path)
        rand_idx = randint(1, len(data) - 101)
        self.mock_X = data.iloc[rand_idx]  # pd.Series
        self.mock_X = self.mock_X.to_dict()
        self.mock_X = Measurement(self.mock_X)
        rand_idx2 = rand_idx + 100
        self.mock_X_batch = data.iloc[rand_idx:rand_idx2]  # pd.DataFrame
        self.mock_X_batch = self.mock_X_batch.to_dict('records')  # list of dicts
        self.mock_X_batch = [Measurement(x) for x in self.mock_X_batch]

        self.input_keys = ['ambient',
                      'coolant',
                      'u_d',
                      'u_q',
                      'motor_speed',
                      'torque',
                      'i_d',
                      'i_q',
                      'stator_yoke',
                      'stator_tooth',
                      'stator_winding']
        self.target_keys = ['pm']

    def tearDown(self):
        os.chdir('..')
        sleep(1)
        shutil.rmtree(self.out_path, ignore_errors=True)

    def test_can_be_spawned(self):
        model = SklearnModelHandler(self.model_path)
        self.assertIs(model.status, ModelStatus.NOT_INITIATED)
        model.spawn()
        self.assertIs(model.status, ModelStatus.READY)

    def test_step(self):
        model = SklearnModelHandler(self.model_path,
                                    input_keys=self.input_keys,
                                    target_keys=self.target_keys)
        model.spawn()
        result = model.step(self.mock_X)
        self.assertIsInstance(result, ModelResponse)

    def test_step_with_incorrect_input_raises_error(self):
        self.input_keys = ['ambient']  # intentionally wrong number of keys

        model = SklearnModelHandler(self.model_path,
                                    input_keys=self.input_keys,
                                    target_keys=self.target_keys)
        model.spawn()
        with self.assertRaises(ModelStepException):
            result = model.step(self.mock_X)

    def test_step_batch(self):
        model = SklearnModelHandler(self.model_path,
                                    input_keys=self.input_keys,
                                    target_keys=self.target_keys)
        model.spawn()
        result = model.step_batch(self.mock_X_batch)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 100)


class MockFmuModelHandlerTests(unittest.TestCase):

    def setUp(self):
        self.out_path = str(uuid.uuid1())
        os.mkdir(self.out_path)
        os.chdir(self.out_path)
        self.fmu_filename = 'CoupledClutches.fmu'
        self.test_file = download_test_file('2.0',
                                            'CoSimulation',
                                            'MapleSim',
                                            '2016.2',
                                            'CoupledClutches',
                                            self.fmu_filename)
        # fmpydump(self.fmu_filename)  # debug
        self.input_keys = ['inputs']
        self.target_keys = ['outputs[1]',
                            'outputs[2]',
                            'outputs[3]',
                            'outputs[4]']
        self.model = FMUModelHandler(self.fmu_filename,
                                     step_size=0.01,
                                     stop_time=1.5,
                                     timestamp_key='index')

        self.model._fmu = MockFMU()

    def tearDown(self):
        os.chdir('..')
        try:
            self.model.destroy()
        except OSError:  # nothing left to destroy
            pass
        sleep(1)
        shutil.rmtree(self.out_path, ignore_errors=True)

    def test_build_response(self):
        exp_res = {'inputs': 1,
                   'outputs[1]': 2,
                   'outputs[2]': 3,
                   'outputs[3]': 4,
                   'outputs[4]': 5}
        res = self.model.step({'index': 1, 'inputs': 1})
        self.assertDictEqual(res, exp_res)


class FmuModelHandlerTests(unittest.TestCase):

    def setUp(self):
        self.out_path = str(uuid.uuid1())
        os.mkdir(self.out_path)
        os.chdir(self.out_path)
        self.fmu_filename = 'CoupledClutches.fmu'
        self.test_file = download_test_file('2.0',
                                            'CoSimulation',
                                            'MapleSim',
                                            '2016.2',
                                            'CoupledClutches',
                                            self.fmu_filename)
        # fmpydump(self.fmu_filename)  # debug
        self.input_keys = ['inputs']
        self.target_keys = ['outputs[1]',
                            'outputs[2]',
                            'outputs[3]',
                            'outputs[4]']
        self.model = FMUModelHandler(self.fmu_filename,
                                step_size=0.01,
                                stop_time=1.5,
                                timestamp_key='index')

    def tearDown(self):
        os.chdir('..')
        try:
            self.model.destroy()
        except OSError:  # nothing left to destroy
            pass
        except AttributeError:  # nothing left to destroy
            pass
        sleep(1)
        shutil.rmtree(self.out_path, ignore_errors=True)

    def test_fmu_model_loaded_succesfully(self):
        self.assertIs(self.model.status, ModelStatus.NOT_INITIATED)
        self.model.spawn()
        self.assertIs(self.model.status, ModelStatus.READY)

    def test_step(self):
        self.model.spawn()

        x = {'index': 0, 'inputs': 0.5}
        expected_res = {'inputs': 0.5,
                        'outputs[2]': 0.09999996829318347,
                        'outputs[3]': 0.0,
                        'outputs[1]': 9.900000031706817,
                        'outputs[4]': 0.0}

        response = self.model.step(x)
        self.assertIsInstance(response, ModelResponse)
        self.assertEqual(len(response.keys()), 5)
        self.assertTrue(response.keys() == expected_res.keys())
        for k in response.keys():
            self.assertAlmostEqual(response[k], expected_res[k])

    def test_multiple_steps(self):
        self.model.spawn()
        for i in range(0, 11):
            x = {'index': i/100, 'inputs': random()}
            response = self.model.step(x)
            self.assertIsInstance(response, ModelResponse)
            self.assertEqual(len(response.keys()), 5)

    def test_get_variable_references(self):
        self.model.spawn()
        self.model.vrs = {}
        self.model._get_value_references()
        expect_in = {'inputs': 17}
        expect_out = {'outputs[1]': 2,
                      'outputs[2]': 4,
                      'outputs[3]': 6,
                      'outputs[4]': 8}
        self.assertDictEqual(self.model.vrs['input'], expect_in)
        self.assertDictEqual(self.model.vrs['output'], expect_out)

    def test_destroy(self):
        self.model.spawn()
        self.model.destroy()
        self.assertFalse(os.path.exists(self.model.unzipdir))

    def test_parse_current_comm_point_initial(self):
        actual = self.model._parse_current_comm_point(X={'index': '2019-01-01 15:00:00'})
        expect = 0
        self.assertEqual(actual, expect)

    def test_parse_current_comm_point(self):
        initial = dt.strptime('2019-01-01 15:00:00', "%Y-%m-%d %H:%M:%S")
        actual = self.model._parse_current_comm_point(X={'index': '2019-01-01 15:03:30'},
                                                      _INITIAL_TIMESTAMP=initial)
        expect = 210
        self.assertEqual(actual, expect)

    def test_parse_current_comm_point_dt_initial(self):
        actual = self.model._parse_current_comm_point(X={'index': dt(2019, 1, 1, 15, 0, 0)})
        expect = 0
        self.assertEqual(actual, expect)

    def test_parse_current_comm_point_dt(self):
        initial = dt.strptime('2019-01-01 15:00:00', "%Y-%m-%d %H:%M:%S")
        actual = self.model._parse_current_comm_point(X={'index': dt(2019, 1, 1, 15, 3, 30)},
                                                      _INITIAL_TIMESTAMP=initial)
        expect = 210
        self.assertEqual(actual, expect)


class ModelHandlerTests(unittest.TestCase):

    def test_add_source(self):
        mh = ModelHandler()
        meas_hand = MeasurementStreamHandler()
        mh.add_source(meas_hand)
        self.assertTrue(len(mh.sources) == 1)
        self.assertEqual(meas_hand, mh.sources[0])

    def test_remove_source(self):
        mh = ModelHandler()
        meas_hand = MeasurementStreamHandler()
        mh.add_source(meas_hand)
        self.assertTrue(len(mh.sources) == 1)
        self.assertEqual(meas_hand, mh.sources[0])
        mh.remove_source(mh.sources[0])
        self.assertTrue(len(mh.sources) == 0)

    def test_pull(self):
        data1 = {'foo' : 1, 'faa' : 2, 'fuu' : 3}
        data2 = {'faa' : 3, 'fyy' : 4, 'fee' : 5}
        mh = ModelHandler()
        meas_hand = MeasurementStreamHandler()
        meas_hand.receive_single(data1) # FIFO 1st
        meas_hand.receive_single(data2) # FIFO 2nd
        mh.add_source(meas_hand)
        meas_hand.add_consumer(mh)

        res = mh.pull()
        self.assertIsInstance(res, list)
        self.assertTrue(len(res) == 1)
        self.assertDictEqual(data1, res[0])

        res = mh.pull()
        self.assertIsInstance(res, list)
        self.assertTrue(len(res) == 1)
        self.assertDictEqual(data2, res[0])

    def test_pull_on_empty_buffer_returns_none(self):
        mh = ModelHandler()
        meas_hand = MeasurementStreamHandler()
        mh.add_source(meas_hand)
        meas_hand.add_consumer(mh)

        res = mh.pull()
        self.assertIsInstance(res, list)
        self.assertTrue(len(res) == 1)
        self.assertIsNone(res[0])

    def test_pull_batch_on_single_source(self):
        mh = ModelHandler()

        meas_hand = MeasurementStreamHandler()
        mh.add_source(meas_hand)
        meas_hand.add_consumer(mh)

        data1 = Measurement({'foo' : 1, 'faa' : 2, 'fuu' : 3})
        data2 = Measurement({'faa' : 3, 'fyy' : 4, 'fee' : 5})
        data3 = Measurement({'foo' : 6, 'fyy' : 7, 'fee' : 8})
        meas_hand.receive_single(data1)  # FIFO 1st
        meas_hand.receive_single(data2)  # FIFO 2nd
        meas_hand.receive_single(data3)  # FIFO 3rd

        res = mh.pull_batch(batch_size=2)  # Excpect [[data1, data2]]
        self.assertTrue(len(res) == 1)
        self.assertTrue(len(res[0]) == 2)
        self.assertIs(res[0][0], data1)
        self.assertIs(res[0][1], data2)


class MeasurementStreamHandlerTests(unittest.TestCase):

    def test_add_consumer(self):
        meas_hand = MeasurementStreamHandler()
        mh = ModelHandler()
        self.assertTrue(len(meas_hand.consumers) == 0)
        meas_hand.add_consumer(consumer=mh)
        self.assertTrue(len(meas_hand.consumers) == 1)
        self.assertIsInstance(meas_hand.consumers[0], ModelHandler)

    def test_remove_consumer(self):
        meas_hand = MeasurementStreamHandler()
        mh = ModelHandler()
        self.assertTrue(len(meas_hand.consumers) == 0)
        meas_hand.add_consumer(consumer=mh)
        self.assertTrue(len(meas_hand.consumers) == 1)
        self.assertIsInstance(meas_hand.consumers[0], ModelHandler)
        meas_hand.remove_consumer(mh)
        self.assertTrue(len(meas_hand.consumers) == 0)

    def test_receive_single(self):
        # TODO
        pass

    def test_receive_batch(self):
        data = [Measurement({'foo': 3, 'faa': 4}), Measurement({'foo': 6, 'faa': 7})]
        meas_hand = MeasurementStreamHandler()
        meas_hand.receive_batch(data)
        self.assertTrue(meas_hand.buffer.qsize() == 2)
        self.assertIs(meas_hand.buffer.get_nowait(), data[0])
        self.assertIs(meas_hand.buffer.get_nowait(), data[1])
        with self.assertRaises(queue.Empty):
            meas_hand.buffer.get_nowait()

    def test_give(self):
        data = [Measurement({'foo': 3, 'faa': 4}), Measurement({'foo': 6, 'faa': 7})]
        meas_hand = MeasurementStreamHandler()
        meas_hand.receive_batch(data)
        self.assertIs(meas_hand.give(), data[0])

    def test_give_batch(self):
        data = [Measurement({'foo': 3, 'faa': 4}),
                Measurement({'foo': 6, 'faa': 7}),
                Measurement({'foo' : 8, 'faa' : 9})]
        meas_hand = MeasurementStreamHandler()
        meas_hand.receive_batch(data)
        batch_size = 2
        res = meas_hand.give_batch(batch_size=batch_size)
        self.assertTrue(len(res) == batch_size)
        self.assertIs(res[0], data[0])
        self.assertIs(res[1], data[1])

    def test_validate_measurement_on_last_datapoint_strategy(self):
        mhand = MeasurementStreamHandler(validation_strategy='LAST_DATAPOINT')
        valid_data = Measurement({'foo': 3, 'faa': 4})
        mhand.last_measurement = valid_data
        data = Measurement({'foo': 14, 'faa': None})
        data = mhand.validate_measurement(data)
        expected_data = ({'foo': 14, 'faa': 4})
        self.assertDictEqual(data, expected_data)

    def test_validate_measurement_on_no_validation_strategy(self):
        mhand = MeasurementStreamHandler(validation_strategy='NO_VALIDATION')
        valid_data = Measurement({'foo': 3, 'faa': 4})
        mhand.last_measurement = valid_data
        data = Measurement({'foo': 14, 'faa': None})
        data = mhand.validate_measurement(data)
        expected_data = ({'foo': 14, 'faa': None})
        self.assertDictEqual(data, expected_data)


class ExperimentTests(unittest.TestCase):

    def setUp(self):
        self.input_keys = ['ambient',
                      'coolant',
                      'u_d',
                      'u_q',
                      'motor_speed',
                      'torque',
                      'i_d',
                      'i_q',
                      'stator_yoke',
                      'stator_tooth',
                      'stator_winding']
        self.target_keys = ['pm']

        self.out_path = str(uuid.uuid1())
        os.mkdir(self.out_path)
        os.chdir(self.out_path)
        self.model_path = '..\\testresources\\pmsm.pickle'
        self.sample_path = '..\\testresources\\pmsm_temperature_sample.csv'
        self.data_path = '..\\testresources\\pmsm_temperature_data.csv'
        conds = [os.path.exists(self.model_path),
                 os.path.exists(self.sample_path),
                 os.path.exists(self.data_path)]
        if not all(conds):
            prepare_dataset()

    def tearDown(self):
        os.chdir('..')
        sleep(1)
        shutil.rmtree(self.out_path, ignore_errors=True)


    def test_initiate_logging(self):
        path = str(uuid.uuid1())
        ex = Experiment()
        log = ex.initiate_logging(path=path)
        self.assertIsInstance(log, TextIOWrapper)
        self.assertFalse(log.closed)
        log.close()
        self.assertTrue(log.closed)
        os.remove(path)

    def test_terminate_logging(self):
        path = str(uuid.uuid1())
        ex = Experiment()
        log = ex.initiate_logging(path=path)
        self.assertIsInstance(log, TextIOWrapper)
        self.assertFalse(log.closed)
        ex.terminate_logging()
        self.assertTrue(log.closed)
        os.remove(path)

    def test_headers_are_written_correctly(self):
        path = str(uuid.uuid1())
        headers = ["meas1", "meas2", "meas3"]
        ex = Experiment()
        log = ex.initiate_logging(path=path, headers=headers)
        self.assertIsInstance(log, TextIOWrapper)
        self.assertFalse(log.closed)
        ex.terminate_logging()
        self.assertTrue(log.closed)
        with open(ex.log_path, 'r') as f:
            self.assertEqual(f.readline(), "meas1,meas2,meas3\n")
        os.remove(path)

    def test_log_row(self):
        path = str(uuid.uuid1())
        ex = Experiment()
        log = ex.initiate_logging(path=path)
        self.assertIsInstance(log, TextIOWrapper)
        self.assertFalse(log.closed)

        row1 = ["1", "2", "3", "4"]
        row2 = ["5", "6", "7", "8"]
        ex.log_row(row1)
        ex.log_row(row2)
        log.close()
        self.assertTrue(log.closed)

        with open(ex.log_path, 'r') as f:
            self.assertIn("1,2,3,4\n", f.readline())
            self.assertIn("5,6,7,8\n", f.readline())

        os.remove(ex.log_path)

    def test_log_row_with_dict_input(self):
        path = str(uuid.uuid1())
        ex = Experiment()
        log = ex.initiate_logging(path=path)
        self.assertIsInstance(log, TextIOWrapper)
        self.assertFalse(log.closed)

        row1 = {'i1': "1", 'o2': "2", 'o3': "3", 'c4': "4"}
        row2 = {'i1': "5", 'o2': "6", 'o3': "7", 'c4': "8"}
        input_keys = ['i1']
        target_keys = ['o2', 'o3']
        control_keys = ['c4']
        model = MockModelHandler(input_keys=input_keys,
                                 target_keys=target_keys,
                                 control_keys=control_keys)
        ex.log_row(row1, model=model)
        ex.log_row(row2, model=model)
        log.close()
        self.assertTrue(log.closed)

        with open(ex.log_path, 'r') as f:
            self.assertIn("1,2,3,4\n", f.readline())
            self.assertIn("5,6,7,8\n", f.readline())

        os.remove(ex.log_path)

    def test_log_row_with_modelresponse_input(self):
        path = str(uuid.uuid1())
        ex = Experiment()
        log = ex.initiate_logging(path=path)
        self.assertIsInstance(log, TextIOWrapper)
        self.assertFalse(log.closed)

        row1 = ModelResponse({'i1': "1", 'o2': "2", 'o3': "3", 'c4': "4"})
        row2 = ModelResponse({'i1': "5", 'o2': "6", 'o3': "7", 'c4': "8"})
        input_keys = ['i1']
        target_keys = ['o2', 'o3']
        control_keys = ['c4']
        model = MockModelHandler(input_keys=input_keys,
                                 target_keys=target_keys,
                                 control_keys=control_keys)
        ex.log_row(row1, model=model)
        ex.log_row(row2, model=model)
        log.close()
        self.assertTrue(log.closed)

        with open(ex.log_path, 'r') as f:
            self.assertIn("1,2,3,4\n", f.readline())
            self.assertIn("5,6,7,8\n", f.readline())

        os.remove(ex.log_path)

    def test_log_row_with_dict_header_input(self):
        path = str(uuid.uuid1())
        ex = Experiment()

        row1 = {'i1': "1", 'o2': "2", 'o3': "3", 'c4': "4"}
        row2 = {'i1': "5", 'o2': "6", 'o3': "7", 'c4': "8"}
        input_keys = ['i1']
        target_keys = ['o2', 'o3']
        control_keys = ['c4']
        headers = input_keys + target_keys + control_keys

        log = ex.initiate_logging(path=path, headers=headers)
        self.assertIsInstance(log, TextIOWrapper)
        self.assertFalse(log.closed)

        model = MockModelHandler(input_keys=input_keys,
                                 target_keys=target_keys,
                                 control_keys=control_keys)
        ex.log_row(row1, model=model)
        ex.log_row(row2, model=model)
        log.close()
        self.assertTrue(log.closed)

        with open(ex.log_path, 'r') as f:
            self.assertIn("i1,o2,o3,c4\n", f.readline())
            self.assertIn("1,2,3,4\n", f.readline())
            self.assertIn("5,6,7,8\n", f.readline())

        os.remove(ex.log_path)

    def test_log_row_with_headers(self):
        path = str(uuid.uuid1())
        headers = ["timestamp", "meas1", "meas2", "meas3"]
        ex = Experiment()
        log = ex.initiate_logging(path=path, headers=headers)
        self.assertIsInstance(log, TextIOWrapper)
        self.assertFalse(log.closed)

        row1 = ["1", "2", "3", "4"]
        row2 = ["5", "6", "7", "8"]
        ex.log_row(row1)
        ex.log_row(row2)
        log.close()
        self.assertTrue(log.closed)

        with open(path, 'r') as f:
            self.assertEqual(f.readline(), "timestamp,meas1,meas2,meas3\n")
            self.assertEqual(f.readline(), "1,2,3,4\n")
            self.assertEqual(f.readline(), "5,6,7,8\n")

        os.remove(ex.log_path)

    def test_add_route(self):
        listen = IncomingMeasurementListener()
        model = ModelHandler()
        ex = Experiment()
        ex.add_route((listen, model))
        self.assertTupleEqual(ex.routes[0], (listen, model))

    def test_add_route_raises_typeerror(self):
        listen = "foo"
        model = ModelHandler()
        ex = Experiment()
        with self.assertRaises(TypeError):
            ex.add_route((listen, model))

    def test_single_source_and_consumer_can_be_added(self):
        listen = IncomingMeasurementListener()
        mdl = ModelHandler()
        ex = Experiment()

        ex.add_route((listen, mdl))

        self.assertEqual(len(ex.routes), 1)
        self.assertListEqual(ex.routes, [(listen, mdl)])

    def test_multiple_setup_calls_yields_warning(self):
        listener = IncomingMeasurementListener()
        model = ModelHandler()
        ex = Experiment()
        ex.add_route((listener, model))
        status = ex.setup()
        self.assertEqual(status, 0)
        status = ex.setup()
        self.assertEqual(status, 1)

    def test_single_source_and_consumer_setup(self):
        listener = IncomingMeasurementListener()
        model = ModelHandler()
        ex = Experiment()
        ex.add_route((listener, model))
        ex.setup()

        self.assertIsInstance(ex.routes[0][0], IncomingMeasurementListener)
        self.assertIsInstance(ex.routes[0][1], ModelHandler)

        self.assertTrue(len(model.sources) == 1)
        self.assertTrue(len(listener.consumers) == 1)
        self.assertTrue(len(ex.routes[0][0].consumers) == 1) # IncomingMeasurementListener
        self.assertTrue(len(ex.routes[0][1].sources) == 1) # ModelHandler

    def test_single_source_and_consumer_run(self):
        data = pd.read_csv("..\\testresources\\pmsm_temperature_sample.csv")

        # Populate test db in memory
        engine = sqla.create_engine('sqlite:///test.db')
        conn = engine.connect()
        data.to_sql('data', con=conn)
        conn.close()

        # source
        qcols = self.input_keys + self.target_keys
        lstnr = IncomingMeasurementPoller(polling_interval=0.1,
                                          db_uri='test.db',
                                          query_cols=qcols)
        # model
        mdel = SklearnModelHandler(model_filename='..\\testresources\\pmsm.pickle',
                                   input_keys=self.input_keys,
                                   target_keys=self.target_keys,
                                   control_keys=self.target_keys)
        exp = OnlineOneToOneExperiment(runtime=0.25)
        exp.add_route((lstnr, mdel))
        exp.setup()
        exp_ok = exp.run()
        self.assertTrue(exp_ok)
        sleep(1)
        os.remove('test.db')

    def test_concurrent_read_write_ops(self):
        # TODO implement timestamp field generalization
        print(os.getcwd())
        mdeel = SklearnModelHandler(model_filename='..\\testresources\\pmsm.pickle',
                                    input_keys=self.input_keys,
                                    target_keys=self.target_keys,
                                    control_keys=self.target_keys)
        # source
        data = pd.read_csv("..\\testresources\\pmsm_temperature_sample.csv")

        engine = sqla.create_engine('sqlite:///test.db')
        conn = engine.connect()

        def simulate_writes():

            def write_row(row: pd.DataFrame):
                row.to_sql('data', con=conn, if_exists='append')

            for _, row in data.iterrows():
                if dt.now() >= exp.stop_time:
                    # Test only, do not use in production
                    # This is only to gracefully stop writing when experiment times out
                    conn.close()
                    break
                write_row(pd.DataFrame(row).transpose())
                sleep(0.1)

        qcols = self.input_keys + self.target_keys
        lstner = IncomingMeasurementPoller(polling_interval=0.5,
                                           db_uri='test.db',
                                           query_cols=qcols,
                                           first_unseen_pk=0)

        log_path = "_test_exp_{}.log".format(str(uuid.uuid1()))
        exp = OnlineOneToOneExperiment(runtime=0.15,
                                       logging=True,
                                       log_path=log_path)
        exp.add_route((lstner, mdeel))
        exp.setup()

        try:
            Thread(target=exp.run).start()
            simulate_writes()
        except Exception:
            raise RuntimeError("Something went horribly wrong")
        finally:
            exp_output = "timestamp," \
                         "ambient," \
                         "coolant," \
                         "u_d," \
                         "u_q," \
                         "motor_" \
                         "speed,torque," \
                         "i_d," \
                         "i_q," \
                         "stator_yoke," \
                         "stator_tooth," \
                         "stator_winding," \
                         "pm," \
                         "pm"

            with open(exp.log_path, 'r') as f:
                self.assertEqual(f.readline().replace('\n', ''), exp_output)
                self.assertGreater(len(f.readline()), 10)
                self.assertGreater(len(f.readline()), 10)
                self.assertGreater(len(f.readline()), 10)
            exp.logger.close()
            os.remove(exp.log_path)
            conn.close()
            sleep(1)
            os.remove('test.db')


class MeasurementTests(unittest.TestCase):

    def test_measurement_can_be_initiated(self):
        meas = Measurement()
        self.assertIsInstance(meas, dict)

    def test_to_numpy(self):
        meas = Measurement({"foo" : 4, "faa" : 6, "fuu" : 7})
        keys = ["fuu", "foo"]
        np_meas = meas.to_numpy(keys=keys)
        self.assertIsInstance(np_meas, np.ndarray)
        np.testing.assert_array_equal(np_meas, np.array([[7, 4]]))


class IncomingMeasurementBatchPollerTests(unittest.TestCase):
    TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

    def setUp(self):
        self.out_path = str(uuid.uuid1())
        os.mkdir(self.out_path)
        os.chdir(self.out_path)
        self.db_path = str(uuid.uuid1()) + ".db"
        self.db_uri = 'sqlite:///{}'.format(self.db_path)
        # I want 7 days of 24 hours with 60 minutes each
        periods = 7 * 24 * 60
        tidx = pd.date_range('2016-07-01', periods=periods, freq='T')

        # Generate random multivariate timeseries and dump to temp sqlite db
        data = pd.DataFrame()
        for var_name in [str(uuid.uuid1()) for _ in range(10)]:
            data[var_name] = pd.Series(data=np.random.randn(periods),
                                       index=tidx)
            data.index.names = ['timestamp']
        engine = create_engine(self.db_uri)  # temp database
        data.to_sql('data', con=engine)

        self.query = "select * from data \n" \
                     "where timestamp >= \"{}\"\n" \
                     "and timestamp < \"{}\""

        self.poller = IncomingMeasurementBatchPoller(self.db_uri,
                                                     self.query,
                                                     90,
                                                     60*60,
                                                     dt(2016, 7, 1, 0, 0, 0),
                                                     dt(2016, 7, 7, 23, 59, 59))

    def tearDown(self):
        os.chdir('..')
        if self.poller.conn and not self.poller.conn.closed:
            self.poller.conn.close()
        sleep(1)
        shutil.rmtree(self.out_path, ignore_errors=True)

    def test_can_be_instantiated(self):
        self.assertIsInstance(self.poller, IncomingMeasurementBatchPoller)

    def test_query_string_is_parsed_correctly_on_path_input(self):
        query_path = str(uuid.uuid1())
        with open(query_path, 'w') as f:
            f.write(self.query)
        self.poller = IncomingMeasurementBatchPoller(
            db_uri=self.db_uri,
            query='..\\testresources\\{}'.format(query_path))
        with open(query_path, 'r') as f:
            self.assertEqual(self.query.replace("\n", ""),
                             f.read().replace("\n", ""))

    def test_can_be_connected_to_database(self):
        self.poller.engine = sqla.create_engine(self.poller.db_uri)
        self.poller.conn = self.poller.engine.connect()
        self.assertFalse(self.poller.conn.closed)

    def test_poll_batch(self):
        self.poller.engine = sqla.create_engine(self.poller.db_uri)
        self.poller.conn = self.poller.engine.connect()
        res = self.poller.poll_batch()
        self.assertIsInstance(res, list)
        self.assertTrue(len(res) == 60)
        for datum in res:
            self.assertIsInstance(datum, dict)

    def test_poll_batch_result_is_added_to_buffer(self):
        self.poller.engine = sqla.create_engine(self.poller.db_uri)
        self.poller.conn = self.poller.engine.connect()
        res = self.poller.poll_batch()
        for meas in res:
             self.assertIs(self.poller.buffer.get_nowait(), meas)
        with self.assertRaises(queue.Empty):
            self.poller.buffer.get_nowait()

    def test_update_timestamps(self):
        new_start, new_stop = self.poller.update_timestamps()
        # old 2016-07-01 00:00:00
        self.assertEqual(new_start, "2016-07-01 01:00:00.000000")
        # old 2016-07-01 00:00:59
        self.assertEqual(new_stop, "2016-07-01 02:00:00.000000")

    def test_update_timestamps_on_exceeding_end(self):
        self.poller = IncomingMeasurementBatchPoller(self.db_uri,
                                                     self.query,
                                                     90,
                                                     60*60,
                                                     dt(2016, 7, 1, 0, 0, 0),
                                                     dt(2016, 7, 1, 1, 30, 00))

        new_start, new_stop = self.poller.update_timestamps()
        self.assertEqual(new_start, "2016-07-01 01:00:00.000000")
        self.assertEqual(new_stop, "2016-07-01 01:30:00.000000")

    def test_update_timestamps_on_exceeding_start(self):
        self.poller = IncomingMeasurementBatchPoller(self.db_uri,
                                                     self.query,
                                                     90,
                                                     60 * 60,
                                                     dt(2016, 7, 1, 0, 0, 0),
                                                     dt(2016, 7, 1, 0, 30, 00))

        with self.assertRaises(ExperimentDurationExceededException):
            new_start, new_stop = self.poller.update_timestamps()

    def test_parse_query_with_path(self):
        rand_dir = str(uuid.uuid1())
        rand_str = str(uuid.uuid1())
        os.mkdir(rand_dir)
        rand_path = './{}/file'.format(rand_dir)
        with open(rand_path, 'w') as f:
            f.write(rand_str)
        self.poller._parse_query(query=rand_path)
        expect = rand_str
        actual = self.poller._parse_query(query=rand_path)
        self.assertEqual(expect, actual)

    def test_parse_query_with_string(self):
        rand_str = str(uuid.uuid1())
        self.poller._parse_query(query=rand_str)
        expect = rand_str
        actual = self.poller._parse_query(query=rand_str)
        self.assertEqual(expect, actual)



    def test_connect(self):
        pass

    def test_receive(self):
        self.poller = \
            IncomingMeasurementBatchPoller(self.db_uri,
                                           self.query,
                                           polling_interval=0.5,
                                           polling_window=3600,
                                           start_time=dt(2016, 7, 1, 0, 0, 0),
                                           stop_time=dt(2016, 7, 1, 23, 59, 59))

        exit_status = self.poller.receive()
        self.assertTrue(exit_status)
        self.assertTrue(self.poller.buffer.qsize() == 24*60)





class IncomingMeasurementListenerTests(unittest.TestCase):

    def test_listen(self):
        listener = IncomingMeasurementListener()

        def invoke_stopevent():
            sleep(15)
            listener._stopevent = Event()

        t = Thread(target=invoke_stopevent, daemon=True)
        t.start()
        exit_status = listener.listen()
        self.assertTrue(exit_status)

    def test_receive_single_json_string(self):
        listener = IncomingMeasurementListener()

        def invoke_stopevent():
            sleep(15)
            listener._stopevent = Event()

        t1 = Thread(target=invoke_stopevent, daemon=True)
        t1.start()

        t2 = Thread(target=listener.listen, daemon=True)
        t2.start()
        message = "{\"var1\": 1, \"var2\": 2, \"var3\": 3}"
        message = "{:<10}".format(str(len(message))) + message
        print(message)

        host = "127.0.0.1"
        port = 33003
        addr = (host, port)

        client_socket = socket(AF_INET, SOCK_STREAM)
        client_socket.connect(addr)
        client_socket.send(bytes(message, "utf8"))
        client_socket.close()
        expected = Measurement({'var1': 1, 'var2': 2, 'var3': 3})
        actual = listener.buffer.get()
        self.assertDictEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
