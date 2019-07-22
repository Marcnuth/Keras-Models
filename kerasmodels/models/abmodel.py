from abc import ABC, abstractmethod



class AbModel(ABC):
    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def build(cls):
        pass
    