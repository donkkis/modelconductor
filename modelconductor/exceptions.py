__package__ = "modelconductor"


class ExperimentDurationExceededException(Exception):
    """Indicates an illegal operation when the experiment has been terminated"""
    pass


class ModelStepException(Exception):
    """Indicates a valid return value was not returned from model"""
    pass
