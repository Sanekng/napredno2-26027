import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
from sklearn.neighbors import KNeighborsClassifier
from framework import (ModelFactory, Accuracy, F1, Observer, IrisExperiment, timed)


class Spy(Observer):
    def __init__(self):
        self.events = []

    def update(self, event, data):
        self.events.append(event)


def test_factory_creates_model():
    assert isinstance(ModelFactory.create("knn", n_neighbors=3), KNeighborsClassifier)


def test_factory_unknown_raises():
    with pytest.raises(ValueError):
        ModelFactory.create("svm")


def test_accuracy_strategy():
    assert Accuracy().compute([0, 1, 1, 0], [0, 1, 0, 0]) == 0.75


def test_f1_strategy_perfect():
    assert F1().compute([0, 1, 2], [0, 1, 2]) == 1.0


def test_observer_receives_events_in_order():
    spy = Spy()
    IrisExperiment("tree", Accuracy(), [spy]).run()
    assert spy.events == ["start", "trained", "finished"]


def test_timed_decorator_records_duration():
    @timed
    def f():
        return 5
    assert f() == 5 and f.last_duration >= 0


@pytest.mark.parametrize("model", ["linear", "tree", "knn"])
def test_iris_demo_scores_well(model):
    assert IrisExperiment(model, Accuracy()).run() > 0.9
