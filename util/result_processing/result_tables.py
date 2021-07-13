DEFAULT_TABLE_ATTRIBUTES = ["seed"]


class DefaultTable:
    STEP = "STEP"
    CUM_TIME = "CUM_TIME"
    LOG_DET_ESTIMATE = "LOG_DET_ESTIMATE"

    def __init__(self):
        self.fields = [self.STEP, self.CUM_TIME, self.LOG_DET_ESTIMATE]
        self.index = self.STEP

    def get_fields(self):
        return self.fields

    def get_index(self):
        return self.index


class DefaultCholeskyTable(DefaultTable):
    DIAGONAL = "DIAGONAL"


class StoppedAlgorithmsTable(DefaultTable):
    UPPER_BOUND = "UPPER_BOUND"

    def __init__(self):
        super().__init__()
        self.fields = self.fields.append(self.UPPER_BOUND)
