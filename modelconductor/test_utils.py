__package__ = "modelconductor"


class MockModelHandler():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class MockFMU:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


    def instantiate(self, *args, **kwargs): pass

    def setupExperiment(self, *args, **kwargs): pass

    def enterInitializationMode(self, *args, **kwargs): pass

    def exitInitializationMode(self, *args, **kwargs): pass

    def setReal(self, *args, **kwargs): pass

    def doStep(self, *args, **kwargs): pass

    def terminate(self, *args, **kwargs): pass

    def freeInstance(self, *args, **kwargs): pass

    def getReal(self, *args, **kwargs):
        # from ModelDescription.xml
        vr_dict = {17: ('inputs', 1),
                   2: ('outputs[1]', 2),
                   4: ('outputs[2]', 3),
                   6: ('outputs[3]', 4),
                   8: ('outputs[4]', 5)}
        return [vr_dict[i][1] for i in args[0]]