"""Mini framework for ML experiments (Factory, Strategy, Observer, Template Method, Decorator)."""
import time
from abc import ABC, abstractmethod
from functools import wraps

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# 1. Factory -------------------------------------------------------------
class ModelFactory:
    _models = {
        "linear": LogisticRegression,
        "tree": DecisionTreeClassifier,
        "knn": KNeighborsClassifier,
    }

    @classmethod
    def create(cls, name, **params):
        if name not in cls._models:
            raise ValueError(f"Unknown model: {name}")
        return cls._models[name](**params)


# 2a. Strategy (metrics) -------------------------------------------------
class Metric(ABC):
    name = "metric"

    @abstractmethod
    def compute(self, y_true, y_pred): ...


class Accuracy(Metric):
    name = "accuracy"

    def compute(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)


class F1(Metric):
    name = "f1"

    def compute(self, y_true, y_pred):
        return f1_score(y_true, y_pred, average="macro")


# 2b. Observer (training monitoring) --------------------------------------
class Observer(ABC):
    @abstractmethod
    def update(self, event, data): ...


class LoggerObserver(Observer):
    def __init__(self):
        self.events = []

    def update(self, event, data):
        self.events.append((event, data))
        print(f"[{event}] {data}")


# 3b. Decorator (timing) --------------------------------------------------
def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        wrapper.last_duration = time.perf_counter() - start
        return result
    wrapper.last_duration = 0.0
    return wrapper


# 3a. Template Method (experiment flow) -----------------------------------
class Experiment(ABC):
    def __init__(self, model_name, metric: Metric, observers=None, **params):
        self.model = ModelFactory.create(model_name, **params)
        self.model_name = model_name
        self.metric = metric
        self.observers = list(observers or [])

    def attach(self, observer):
        self.observers.append(observer)

    def notify(self, event, data):
        for o in self.observers:
            o.update(event, data)

    @abstractmethod
    def load_data(self): ...

    @timed
    def train(self, X, y):
        self.model.fit(X, y)

    def evaluate(self, X, y):
        return self.metric.compute(y, self.model.predict(X))

    def run(self):  # the template method: fixed order of steps
        X_tr, X_te, y_tr, y_te = self.load_data()
        self.notify("start", self.model_name)
        self.train(X_tr, y_tr)
        self.notify("trained", f"{self.train.last_duration:.4f}s")
        score = self.evaluate(X_te, y_te)
        self.notify("finished", f"{self.metric.name}={score:.3f}")
        return score


class IrisExperiment(Experiment):
    def load_data(self):
        X, y = load_iris(return_X_y=True)
        return train_test_split(X, y, test_size=0.3, random_state=42)


if __name__ == "__main__":
    for m in ("linear", "tree", "knn"):
        IrisExperiment(m, Accuracy(), [LoggerObserver()]).run()
