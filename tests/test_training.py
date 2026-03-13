import pytest
from mnist_repo.data import corrupt_mnist 
from mnist_repo.train import train
#from mnist_repo.model import MyAwesomeModel 


def test_train(mocker):
    #Arrange
    mock_dataset = mocker.patch("mnist_repo.data.corrupt_mnist")
    mock_dataset.return_value = corrupt_mnist()

    #Act
    train(epochs=0)

    #Assert
    assert mock_dataset.call_count == 1 #single handedly the most painful experience to figure out

def test_train_negative_learning_rate():
    #Arrange
    lr = -1.0
    epochs = 0

    #Act
    with pytest.raises(ValueError):
    
    #Assert
        train(lr=lr,epochs=epochs)

#def test_train_one_iteration(mocker):
    #Arrange
    #mock_dataset = mocker.patch("mnist_repo.data.corrupt_mnist")
    #mock_dataset.return_value = corrupt_mnist()

    #mock_model = mocker.Mock(spec=MyAwesomeModel)
    #Act
    #train(epochs=1)

    #Assert
    #assert mock_model.train.assert_called_once()
