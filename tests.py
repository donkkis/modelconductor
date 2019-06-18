import pickle
import queue
import unittest
import sqlalchemy
from handlers import SklearnModelHandler, FMUModelHandler, IncomingMeasurementListener
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
from datetime import datetime as dt

class SklearnTests(unittest.TestCase):

    def test_iterate_and_plot(self):
        with open('..\\src\\nox_idx.pickle', 'rb') as pickle_file:
            idx = pickle.load(pickle_file)

        with open('..\\src\\data.pickle', 'rb') as pickle_file:
            data = pickle.load(pickle_file)

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
        data = [{'foo': 3, 'faa': 4}, {'foo': 6, 'faa': 7}]
        meas_hand = MeasurementStreamHandler()
        meas_hand.receive_batch(data)
        self.assertTrue(meas_hand.buffer.qsize() == 2)
        self.assertIs(meas_hand.buffer.get_nowait(), data[0])
        self.assertIs(meas_hand.buffer.get_nowait(), data[1])
        with self.assertRaises(queue.Empty):
            meas_hand.buffer.get_nowait()


class ExperimentTests(unittest.TestCase):

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

        exp = OnlineOneToOneExperiment(runtime=0.5)
        exp.add_route((lstnr, mdel))
        exp.setup()
        exp_ok = exp.run()
        self.assertTrue(len(exp.results) == 51)
        self.assertTrue(exp_ok)


    def test_concurrent_read_write_ops(self):
        # consumer
        with open('..\\src\\nox_idx.pickle', 'rb') as f:
            qcolss = pickle.load(f)
            qcolss_string = ', '.join('"{0}"'.format(qcol) for qcol in qcolss)

        mdeel = SklearnModelHandler(model_filename='..\\src\\nox_rfregressor.pickle', input_keys=qcolss)

        # source
        data = pd.read_csv("..\\src\\NRTC_laskenta\\Raakadata_KAIKKI\\nrtc1_ref_10052019.csv", delimiter=";")
        print(data.head())

        def simulate_writes():
            engine = sqla.create_engine('sqlite:///test.db')
            conn = engine.connect()

            def write_row(row: pd.DataFrame):
                row.to_sql('data', con=conn, if_exists='append')

            for _, row in data.iterrows():
                write_row(pd.DataFrame(row).transpose())
                sleep(0.1)

        lstner = IncomingMeasurementPoller(polling_interval=0.5,
                                          db_uri='test.db',
                                          query_cols=qcolss_string,
                                          first_unseen_pk=0)

        exp = OnlineOneToOneExperiment(runtime=0.5)
        exp.add_route((lstner, mdeel))
        exp.setup()

        threading.Thread(target=exp.run).start()
        threading.Thread(target=simulate_writes).start()





if __name__ == '__main__':
    unittest.main()