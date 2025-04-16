from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
import numpy as np


def edited_nearest_neighbors(x_train: np.array, y_train: np.array, n_neighbors: int):
    """
    ENN using imblearn with training data input.

    :param x_train:
    :param y_train:
    :param n_neighbors:
    :return:
    """
    enn = EditedNearestNeighbours(n_neighbors=n_neighbors)
    x_train_enn, y_train_enn = enn.fit_resample(x_train, y_train)

    return x_train_enn, y_train_enn


def random_undersampling(x_train: np.array, y_train: np.array):
    """

    :param x_train:
    :param y_train:
    :return:
    """
    ru = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    x_train_ru, y_train_ru = ru.fit_resample(x_train, y_train)

    return x_train_ru, y_train_ru


def random_oversampling(x_train: np.array, y_train: np.array):
    """

    :param x_train:
    :param y_train:
    :return:
    """
    ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
    x_train_ro, y_train_ro = ros.fit_resample(x_train, y_train)

    return x_train_ro, y_train_ro


def smote_sampling(x_train: np.array, y_train: np.array):
    """

    :param x_train:
    :param y_train:
    :return:
    """
    smote = SMOTE(k_neighbors=5, random_state=689)
    x_train_sm, y_train_sm = smote.fit_resample(x_train, y_train)

    return x_train_sm, y_train_sm


def tomelinks_sampling(x_train: np.array, y_train: np.array):
    """

    :param x_train:
    :param y_train:
    :return:
    """
    tl = TomekLinks()
    x_train_tl, y_train_tl = tl.fit_resample(x_train, y_train)

    return x_train_tl, y_train_tl


def importance_sampling(x_train: np.array, y_train: np.array):
    """
    Importance Sampling returns the original dataset but with sample weights.

    :param x_train: Training data
    :param y_train: Labels
    :return: x_train, y_train, sample_weights
    """

    def compute_class_weights(y_train: np.array):
        """
        Compute inverse class frequency weights.

        :param y_train: Binary class labels (0/1)
        :return: Dictionary of weights per class
        """
        unique, counts = np.unique(y_train, return_counts=True)
        total = sum(counts)
        class_weights = {cls: total / count for cls, count in zip(unique, counts)}
        return class_weights

    class_weights = compute_class_weights(y_train)
    sample_weights = np.array([class_weights[y] for y in y_train])
    return x_train, y_train, sample_weights

