class FailedToConverge(Exception):
    def __init__(self):
        self.message = 'Gradient Descent failed to converge'
