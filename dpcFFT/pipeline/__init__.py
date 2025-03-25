"""
Pipeline class
"""


class Pipeline:
    def __init__(self, *args):
        self.steps = args

    def __call__(self, data):
        for step in self.steps:
            data = step(data)
        return data
