class FailedToConverge(Exception):
    def __init__(self):
        self.message = 'Gradient Descent failed to converge.\n' \
                       'Most probably this is because your alpha parameter is too large.\n' \
                       'Try to set a smaller value for alpha.\n' \
                       'Note: alpha should be strictly higher than 0.\n' \
                       'P.S. You might also limit the initial range of values by setting the parameter _range.\n'

        self.__cause__ = RuntimeWarning()
