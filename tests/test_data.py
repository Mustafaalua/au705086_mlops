from torch.utils.data import Dataset
from mnist_repo.data import MyDataset
from mnist_repo.data import corrupt_mnist
import torch

def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MyDataset("data/raw")
    assert isinstance(dataset, Dataset)


def test_data():
    #Arrange
    expected_train_size = 30000
    expected_test_size = 5000
    expected_input_dim = (1, 28, 28)
    expected_labels = range(10)
    expected_tensor_labels = torch.arange(0,10)

    #Act
    train, test = corrupt_mnist()

    #Assert
    assert len(train) == expected_train_size
    assert len(test) == expected_test_size
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == expected_input_dim
            assert y in expected_labels

    train_targets = torch.unique(train.tensors[1])
    test_targets = torch.unique(test.tensors[1])

    assert torch.all(train_targets == expected_tensor_labels)
    assert torch.all(test_targets == expected_tensor_labels)
