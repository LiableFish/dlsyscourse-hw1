import struct
import gzip
from typing import Optional, Tuple

import numpy as np

import sys


sys.path.append('python/')
import needle as ndl


def _min_max_scaler(
    data: np.ndarray,
    *,
    max_range: float,
    min_range: float,
    axis: Optional[Tuple[int]] = None,
):
    """inspired by sklearn"""
    std = (data - data.min(axis=axis)) / (data.max(axis=axis) - data.min(axis=axis))
    return std * (max_range - min_range) + min_range


def _read_mnist_images(image_filename: str) -> np.ndarray:
    with gzip.open(image_filename, 'rb') as file:
        _, number_of_images, number_of_rows, number_of_cols = struct.unpack(">4I", file.read(16))
        data = np.frombuffer(file.read(), dtype=np.dtype(np.uint8).newbyteorder('>')).astype(np.float32)
        data = data.reshape(number_of_images, number_of_rows * number_of_cols)
        return _min_max_scaler(data, min_range=0, max_range=1)


def _read_mnist_labels(label_filename: str) -> np.ndarray:
    with gzip.open(label_filename, 'rb') as file:
        file.read(8)  # skip magic number and number of labels
        labels = np.frombuffer(file.read(), dtype=np.dtype(np.uint8).newbyteorder('>'))
        return labels


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    return _read_mnist_images(image_filename), _read_mnist_labels(label_filename)
    ### END YOUR CODE


def softmax_loss(Z: ndl.Tensor, y_one_hot: ndl.Tensor):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    batch_size = Z.shape[0]
    return ndl.ops.summation(
        ndl.ops.log(ndl.ops.exp(Z).sum(axes=(1,))) - (Z * y_one_hot).sum(axes=(1,)),
    ) / batch_size


def _onehot(y: np.ndarray, number_of_classes: int) -> np.ndarray:
    return np.identity(number_of_classes)[y]

def nn_epoch(
    X: np.ndarray,
    y: np.ndarray,
    W1: ndl.Tensor,
    W2: ndl.Tensor,
    lr: float = 0.1,
    batch: int = 100,
):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """
    y_onehot = _onehot(y, number_of_classes=W2.shape[1])

    for start in range(0, X.shape[0], batch):
        X_batch = ndl.Tensor(X[start:start + batch])
        y_batch = ndl.Tensor(y_onehot[start:start + batch], required_grad=False)
        logits = ndl.ops.relu(X_batch @ W1) @ W2
        loss = softmax_loss(logits, y_batch)
        loss.backward()

        W1 = ndl.Tensor(W1.numpy() - lr * W1.grad.numpy())
        W2 = ndl.Tensor(W2.numpy() - lr * W2.grad.numpy())

    return W1, W2


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
