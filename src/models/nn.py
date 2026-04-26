import numpy as np
import torch.nn as nn
from skorch import NeuralNetClassifier


def neural_net(X_train, y_train):
    n_input = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    module = nn.Sequential(
        nn.Linear(n_input, 128),
        nn.ReLU(),
        nn.Linear(128, n_classes),
    )
    clf = NeuralNetClassifier(
        module,
        max_epochs=20,
        lr=1e-3,
        batch_size=64,
        device='cpu',
        verbose=0,
    )
    clf.fit(
        X_train.astype(np.float32),
        y_train.astype(np.int64),
    )
    return clf
