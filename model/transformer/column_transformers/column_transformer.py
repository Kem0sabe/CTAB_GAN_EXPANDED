from abc import ABC, abstractmethod

class Column_transformer(ABC):

    def __init__(self):
        self.model = None
        self.components = None
        self.output_info = None
        self.output_dim = 0
        self.ordering = None

    def get_model(self):
        return self.model

    def get_component(self):
        return self.components

    def get_output_info(self):
        return self.output_info

    def get_output_dim(self):
        return self.output_dim

    def get_ordering(self):
        return self.ordering
        
    @abstractmethod
    def fit(self, data_col):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def transform(self, data_col,st):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def inverse_transform(self, data):
        raise NotImplementedError("Subclasses must implement this method")
  