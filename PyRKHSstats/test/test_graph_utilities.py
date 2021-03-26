import pytest
import numpy as np

from graph_utilities import validate_adjacency_matrix, \
    validate_directed_graph, \
    validate_directed_acyclic_graph


@pytest.mark.parametrize(
    "matrix,expected",
    [
        (np.asarray([]), False),
        (np.asarray([1, 0, 1]), False),
        (np.asarray([[1, 0, 1], [1, 0, 0]]), False),
        (np.asarray([[1, 1], [1, 1]]), True),
        (np.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), True),
        (np.asarray([[1, 0, 0], [1, 2, 1], [1, 0, 1]]), False),
    ]
)
def test_validate_adjacency_matrix(matrix, expected):

    assert validate_adjacency_matrix(matrix) == expected


@pytest.mark.parametrize(
    "matrix,expected",
    [
        (np.asarray([]), False),
        (np.asarray([1, 0, 1]), False),
        (np.asarray([[1, 0, 1], [1, 0, 0]]), False),
        (np.asarray([[1, 1], [1, 1]]), False),
        (np.asarray([[1, 0, 0], [1, 2, 1], [1, 0, 1]]), False),
        (np.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), True),
        (np.asarray([[1, 0, 0], [0, 0, 0], [0, 0, 0]]), False),
        (np.asarray([[0, 0, 1], [0, 0, 0], [1, 0, 0]]), False),
        (np.asarray([[0, 1, 0], [0, 0, 1], [1, 0, 0]]), True),
    ]
)
def test_validate_directed_graph(matrix, expected):

    assert validate_directed_graph(matrix) == expected


@pytest.mark.parametrize(
    "matrix,expected",
    [
        (np.asarray([]), False),
        (np.asarray([1, 0, 1]), False),
        (np.asarray([[1, 0, 1], [1, 0, 0]]), False),
        (np.asarray([[1, 1], [1, 1]]), False),
        (np.asarray([[1, 0, 0], [1, 2, 1], [1, 0, 1]]), False),
        (np.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), True),
        (np.asarray([[1, 0, 0], [0, 0, 0], [0, 0, 0]]), False),
        (np.asarray([[0, 0, 1], [0, 0, 0], [1, 0, 0]]), False),
        (np.asarray([[0, 1, 0], [0, 0, 1], [1, 0, 0]]), False),
        (np.asarray([[0, 1, 0], [0, 0, 1], [0, 0, 0]]), True)
    ]
)
def test_validate_directed_acyclic_graph(matrix, expected):

    assert validate_directed_acyclic_graph(matrix) == expected
