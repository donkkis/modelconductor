import pickle
import queue
import unittest
from io import TextIOWrapper
import os
import numpy as np
import sqlalchemy
from keras.layers import Dense, Activation
from handlers import SklearnModelHandler, FMUModelHandler, IncomingMeasurementListener, KerasModelHandler, Measurement, \
    OnlineBatchTrainableExperiment
from handlers import Experiment
from handlers import ModelHandler
from handlers import IncomingMeasurementPoller
from handlers import OnlineOneToOneExperiment
from handlers import MeasurementStreamHandler
from handlers import MeasurementStreamPoller
from handlers import IncomingMeasurementBatchPoller
from matplotlib import pyplot as plt
import sqlalchemy as sqla
import pandas as pd
from time import sleep
import os
import threading
from datetime import datetime as dt, timedelta
import pprint
import uuid

# TODO cleanup generated dummy log and db files after testing

class SklearnTests(unittest.TestCase):

    def test_iterate_and_plot(self):
        with open('..\\src\\nox_idx.pickle', 'rb') as pickle_file:
            idx = pickle.load(pickle_file)

        with open('..\\src\\data.pickle', 'rb') as pickle_file:
            data = pd.read_pickle(pickle_file)

        X = data[idx].to_numpy()[600:625]
        y = data["Left_NOx"].to_numpy()[600:625]

        sk_hand = SklearnModelHandler(
            model_filename='..\\src\\nox_rfregressor.pickle')

        def step_iterate(X, y):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            x1 = []
            y1 = []
            x2 = []
            y2 = []
            plt.ion()
            plt.show()
            for i in range(X.shape[0]):
                xrow = X[i, :]
                yrow = y[i]
                result = list(sk_hand.model.predict(xrow.reshape(1, -1)))
                # print(result[0], yrow)

                # plotting demo
                x1.append(i / 10)
                x2.append(i / 10)
                y1.append(yrow)
                y2.append(result[0])
                ax.clear()
                ax.plot(x1, y1)
                ax.plot(x2, y2)
                plt.draw()
                plt.legend(['NOx, measured [ppm]', 'Nox simulated [ppm]'])
                plt.title('Engine-out NOx vs NRTC Cycle')
                plt.xlabel('Time (s)')
                plt.pause(0.001)

        step_iterate(X, y)
        sk_hand.destroy()


class FmuTests(unittest.TestCase):
    def some_test(self):
        fmu_hand = FMUModelHandler(
            fmu_filename='C:\\Users\\paho\\Dropbox\\Opiskelu\\Diplomityö\\src\\fmi_simulink_demo\\compute_power_5_2.fmu',
            start_time=0.0,
            threshold=2.0,
            stop_time=1238,
            step_size=1
        )


class IncominmgMeasurementTests(unittest.TestCase):

    def some_test(self):
        meas_hand = IncomingMeasurementListener()


class ModelHandlerTests(unittest.TestCase):

    def test_add_source(self):
        mh = ModelHandler()
        meas_hand = MeasurementStreamHandler()
        mh.add_source(meas_hand)
        self.assertTrue(len(mh.sources) == 1)
        self.assertEqual(meas_hand, mh.sources[0])

    def test_add_source_raises_typeerror(self):
        mh = ModelHandler()
        with self.assertRaises(TypeError):
            mh.add_source("jou")

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


class KerasModelHandlerTests(unittest.TestCase):

    def test_keras_modelhandler_can_be_initiated_from_scratch(self):
        layers = [
            Dense(32, input_shape=(784,)),
            Activation('relu'),
            Dense(10),
            Activation('softmax')]
        model = KerasModelHandler(layers=layers)
        self.assertIsInstance(model, KerasModelHandler)

    def test_keras_model_can_be_spawned(self):
        layers = [
            Dense(32, input_shape=(784,)),
            Activation('relu'),
            Dense(10),
            Activation('softmax')]
        model = KerasModelHandler(layers=layers)
        model.spawn()
        self.assertIsInstance(model, KerasModelHandler)
        self.assertEqual("Ready", model.status)

    def test_fit_single(self):
        layers = [
            Dense(9, input_shape=(3,)),
            Activation('relu'),
            Dense(1)]

        meas = Measurement({"foo" : 3, "faa" : 4, "fuu" : 5, "fee" : 6})
        input_keys = ["foo", "fee", "fuu"]
        target_keys = ["faa"]

        model = KerasModelHandler(layers=layers,
                                  input_keys=input_keys,
                                  target_keys=target_keys)
        model.spawn()
        model.fit(measurement=meas)
        self.assertEqual("Ready", model.status)

    def test_measurements_to_numpy(self):
        layers = [
            Dense(9, input_shape=(3,)),
            Activation('relu'),
            Dense(1)]

        meas = Measurement({"foo" : 3, "faa" : 4, "fuu" : 5, "fee" : 6})
        meas2 = Measurement({"foo" : 12, "faa" : 3, "fuu" : 32, "fee" : 56})
        meas3 = Measurement({"foo" : 43, "faa" : 23, "fuu" : 76, "fee" : 64})
        meas4 = Measurement({"foo" : 1, "faa" : 12, "fuu" : 86, "fee" : 23})
        input_keys = ["foo", "fee", "fuu"]
        target_keys = ["faa"]

        model = KerasModelHandler(layers=layers,
                                  input_keys=input_keys,
                                  target_keys=target_keys)
        model.spawn()
        X, y = model.measurements_to_numpy(measurements=[meas, meas2, meas3, meas4])
        # print(X.shape)  # debug (4, 3)
        # print(y.shape)  # debug (4, 1)
        self.assertTupleEqual(X.shape, (4, 3))
        self.assertTupleEqual(y.shape, (4, 1))
        X_true = np.array([
            [3, 6, 5],
            [12, 56, 32],
            [43, 64, 76],
            [1, 23, 86]
        ])
        y_true = np.array([[4], [3], [23], [12]])
        np.testing.assert_array_equal(X, X_true)
        np.testing.assert_array_equal(y, y_true)

    def test_fit_batch(self):
        layers = [
            Dense(9, input_shape=(3,)),
            Activation('relu'),
            Dense(1)]

        meas = Measurement({"foo" : 3, "faa" : 4, "fuu" : 5, "fee" : 6})
        meas2 = Measurement({"foo" : 12, "faa" : 3, "fuu" : 32, "fee" : 56})
        meas3 = Measurement({"foo" : 43, "faa" : 23, "fuu" : 76, "fee" : 64})
        meas4 = Measurement({"foo" : 1, "faa" : 12, "fuu" : 86, "fee" : 23})
        input_keys = ["foo", "fee", "fuu"]
        target_keys = ["faa"]

        model = KerasModelHandler(layers=layers,
                                  input_keys=input_keys,
                                  target_keys=target_keys)
        model.spawn()
        measurements = [meas, meas2, meas3, meas4]
        weights_pre = model.model.get_weights()  # List[np.ndarray]
        for i in range(len(weights_pre)):
            np.testing.assert_array_equal(weights_pre[i], model.model.get_weights()[i])

        model.fit_batch(measurements)
        self.assertEqual("Ready", model.status)

    def test_step_fit_batch(self):
        layers = [
            Dense(9, input_shape=(3,)),
            Activation('relu'),
            Dense(1)]

        meas = Measurement({"foo" : 3, "faa" : 4, "fuu" : 5, "fee" : 6})
        meas2 = Measurement({"foo" : 12, "faa" : 3, "fuu" : 32, "fee" : 56})
        meas3 = Measurement({"foo" : 43, "faa" : 23, "fuu" : 76, "fee" : 64})
        meas4 = Measurement({"foo" : 1, "faa" : 12, "fuu" : 86, "fee" : 23})
        input_keys = ["foo", "fee", "fuu"]
        target_keys = ["faa"]

        model = KerasModelHandler(layers=layers,
                                  input_keys=input_keys,
                                  target_keys=target_keys)
        model.spawn()
        measurements = [meas, meas2, meas3, meas4]
        _, results = model.step_fit_batch(measurements=measurements)
        self.assertTrue(len(results) == 4)

    def test_fit_batch_learning(self):
        pass
        """
        weights_post = model.model.get_weights()
        for i in range(len(weights_post)):
            np.testing.assert_array_equal(weights_post[i], model.model.get_weights()[i])

        with self.assertRaises(Exception):
            for i in range(len(weights_pre)):
                np.testing.assert_array_equal(weights_pre[i], weights_post[i])

        print(model.model.total_loss)
        print(model.model.to_json())
        print(model.model.to_yaml())
        """

class ExperimentTests(unittest.TestCase):

    def test_initiate_logging(self):
        tic = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = "experiment_{}.log".format(tic)
        ex = Experiment()
        log = ex.initiate_logging(path=path)
        self.assertIsInstance(log, TextIOWrapper)
        self.assertFalse(log.closed)
        log.close()
        self.assertTrue(log.closed)
        os.remove(path)

    def test_terminate_logging(self):
        tic = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = "experiment_{}.log".format(tic)
        ex = Experiment()
        log = ex.initiate_logging(path=path)
        self.assertIsInstance(log, TextIOWrapper)
        self.assertFalse(log.closed)
        ex.terminate_logging()
        self.assertTrue(log.closed)
        os.remove(path)

    def test_headers_are_written_correctly(self):
        tic = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = "experiment_{}.log".format(tic)
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
        tic = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = "experiment_{}.log".format(tic)
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
            self.assertEqual(f.readline(), "1,2,3,4\n")
            self.assertEqual(f.readline(), "5,6,7,8\n")

        os.remove(ex.log_path)

    def test_log_row_with_headers(self):
        tic = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = "experiment_{}.log".format(tic)
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
        with open('..\\src\\nox_idx.pickle', 'rb') as f:
            qcols = pickle.load(f)
            qcols_string = ', '.join('"{0}"'.format(qcol) for qcol in qcols)

        lstnr = IncomingMeasurementPoller(polling_interval=0.1,
                                          db_uri='..\\src\\data_small.db',
                                          query_cols=qcols_string,
                                          first_unseen_pk=5000)
        mdel = SklearnModelHandler(model_filename='..\\src\\nox_rfregressor.pickle', input_keys=qcols)

        # print("now", dt.now())  #  debug
        exp = OnlineOneToOneExperiment(runtime=0.5)
        # print("start", exp.start_time)  # debug
        # print("stop", exp.stop_time)  # debug
        exp.add_route((lstnr, mdel))
        exp.setup()
        exp_ok = exp.run()
        # print(len(exp.results))  # debug
        self.assertTrue(len(exp.results) == 51)
        self.assertTrue(exp_ok)

    def test_concurrent_read_write_ops(self):
        # consumer
        with open('..\\src\\nox_idx.pickle', 'rb') as f:
            qcolss = pickle.load(f)
            qcolss_string = ', '.join('"{}"'.format(qcol) for qcol in qcolss)

        qcolss_string += ', "Left_NOx", "Time"'

        # TODO implement timestamp field generalization
        mdeel = SklearnModelHandler(model_filename='..\\src\\nox_rfregressor.pickle',
                                    input_keys=qcolss,
                                    target_keys=["Left_NOx"])

        # source
        data = pd.read_csv("..\\src\\NRTC_laskenta\\Raakadata_KAIKKI\\nrtc1_ref_10052019.csv", delimiter=";")
        print(data.head())

        def simulate_writes():
            engine = sqla.create_engine('sqlite:///test.db')
            conn = engine.connect()

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

        lstner = IncomingMeasurementPoller(polling_interval=0.5,
                                          db_uri='test.db',
                                          query_cols=qcolss_string,
                                          first_unseen_pk=0)

        exp = OnlineOneToOneExperiment(runtime=0.5,
                                       logging=True,
                                       log_path="_test_exp_{}.log".format(str(uuid.uuid1())))
        exp.add_route((lstner, mdeel))
        exp.setup()

        try:
            threading.Thread(target=exp.run).start()
            simulate_writes()
        except Exception:
            raise RuntimeError("Something went horribly wrong")
        finally:
            with open(exp.log_path, 'r') as f:
                self.assertEqual(f.readline(), "timestamp,Left_NOx_meas,Left_NOx_pred\n")
                self.assertGreater(len(f.readline()), 10)
                self.assertGreater(len(f.readline()), 10)
                self.assertGreater(len(f.readline()), 10)

class OnlineBatchTrainableExperimentTests(unittest.TestCase):

    def test_log_batch(self):
        path = '{}.log'.format(str(uuid.uuid1()))
        ex = OnlineBatchTrainableExperiment(log_path=path)

        ts1 = str(dt.now())
        ts2 = str(dt.now() + timedelta(seconds=5))
        ts3 = str(dt.now() + timedelta(seconds=10))
        ts4 = str(dt.now() + timedelta(seconds=15))

        meas1 = Measurement({"timestamp": ts1, "faa": 4, "fuu": 5, "fee": 6})
        meas2 = Measurement({"timestamp": ts2, "faa": 65, "fuu": 323, "fee": 123})
        meas3 = Measurement({"timestamp": ts3, "faa": 43, "fuu": 43, "fee": 654})
        meas4 = Measurement({"timestamp": ts4, "faa": 12, "fuu": 56, "fee": 78})
        result = [44, 34, 76, 98]
        timestamp_key = "timestamp"
        groundtruth_key = "fee"
        data = [meas1, meas2, meas3, meas4]
        headers = ["Timestamp", "Ground Truth", "Prediction"]
        ex.initiate_logging(path=ex.log_path, headers=headers)
        ex.log_batch(data, groundtruth_key, timestamp_key, result, debug=True)
        ex.terminate_logging()

        with open(path, 'r') as f:
            self.assertEqual(f.readline(), 'Timestamp,Ground Truth,Prediction\n')
            self.assertEqual(f.readline(), ts1 + ',6,44\n')
            self.assertEqual(f.readline(), ts2 + ',123,34\n')
            self.assertEqual(f.readline(), ts3 + ',654,76\n')
            self.assertEqual(f.readline(), ts4 + ',78,98\n')

        os.remove(path)

    def test_run(self):
        # TODO Fix the test so it terminates in finite time
        # consumer
        with open('..\\src\\nox_idx.pickle', 'rb') as f:
            qcols = pickle.load(f)

        ex = OnlineBatchTrainableExperiment(batch_size=60,logging=True)

        start_time = dt(2019, 5, 10, 11, 18, 33)
        stop_time = dt(2019, 5, 10, 12, 18, 33)

        poller = IncomingMeasurementBatchPoller(db_uri='sqlite:///..\\src\\data.db',
                                                query_path='data_query',
                                                polling_interval=1,
                                                polling_window=1,
                                                start_time=start_time,
                                                stop_time=stop_time)

        layers = [
            Dense(128, input_shape=(len(qcols),)),
            Activation('relu'),
            Dense(64),
            Activation('relu'),
            Dense(32),
            Activation('relu'),
            Dense(1)]

        mhand = KerasModelHandler(input_keys=qcols,
                                  target_keys=["Left_NOx"],
                                  layers=layers)

        ex.add_route((poller, mhand))
        ex.setup()
        ex.run()

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


if __name__ == '__main__':
    unittest.main()