from abc import ABC
from typing import Self

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ModelConfigurations:
    """
    Base class for model configuration.
    This class is used to define the configuration for a model.
    """
    pass

class BaseModel(BaseEstimator):

    def fit(self, X, y=None, *, fit_params: ModelConfigurations | None=None) -> Self:
        """
        Fit the model to the data.

        :param X: The input data.
        :param y: The target variable (not used).
        :param fit_params: The fit configurations for the model.
        :type fit_params: :class:`~recomease.models.base_model.ModelConfigurations` | None
        :return: self
        :rtype: Self
        """
        pass

    def predict(self, X, *, predict_params: ModelConfigurations | None=None) -> np.ndarray:

        """
        Predict the output for the given input data.

        :param X: The input data.
        :param predict_params: The predict configurations for the model.
        :type predict_params: :class:`~recomease.models.base_model.ModelConfigurations` | None
        :return: The predicted output.
        :rtype: np.ndarray
        """
        pass

    def get_params(self, deep: bool=True) -> dict:
        """
        Get the parameters of the model.

        :param deep: If True, get parameters of the model and its sub-models.
        :type deep: bool
        :return: A dictionary of parameters.
        :rtype: dict
        """
        pass

    def set_params(self, **params: dict) -> Self:
        """
        Set the parameters of the model.

        :param params: The parameters to set.
        :type params: dict
        :return: self
        :rtype: Self
        """
        pass