import torch
from mnist_repo.model import MyAwesomeModel
import pytest


def test_model():
    # Arrange
    expected_output_shape = (1, 10)
    input = torch.zeros(1, 1, 28, 28)
    model = MyAwesomeModel()

    # Act
    output = model(input)

    # Assert
    assert output.shape == expected_output_shape


@pytest.mark.parametrize("batch_size", [32, 64])
def test_model_batch_size(batch_size):
    # Arrange
    expected_output_shape = (batch_size, 10)
    input = torch.zeros(batch_size, 1, 28, 28)
    model = MyAwesomeModel()

    # Act
    output = model(input)

    # Assert
    assert output.shape == expected_output_shape
