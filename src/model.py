from abc import ABC, abstractmethod


class Model(ABC):

    def __init__(self, model_type, do_train) -> None:
        self.model_type = model_type
        self.do_train = do_train

    @abstractmethod
    def inference(self):
        raise NotImplemented

    @abstractmethod
    def get_loss(self) -> float:
        if do_train:
            raise NotImplemented
        else:
            pass

    @property
    @abstractmethod
    def insert_data(self):
        raise NotImplemented

    @abstractmethod
    def visualize(self):
        raise NotImplemented

    @abstractmethod
    def train(self):
        if do_train:
            raise NotImplemented
        else:
            pass

    @abstractmethod
    def validate(self):
        if do_train:
            raise NotImplemented
        else:
            pass

    @abstractmethod
    def save(self):
        raise NotImplemented

    def get_model_type(self):
        return self.model_type




