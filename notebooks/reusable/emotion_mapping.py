from enum import Enum

import numpy as np
from sklearn.base import clone


class EmotionMap(Enum):
    """
        Mapping of emotions into two components, valence and arousal.
        Based on an approximation from https://www.researchgate.net/figure/Emotion-Mapping-in-Valence-Arousal-Domains_fig2_348990229
    """
    Anger = (-0.75, 0.25)
    Disgust = (-0.25, -0.5)
    Fear = (-0.25, 0.75)
    Happy = (1, 0.25)
    Neutral = (0, 0)
    Sad = (-1, -0.5)
    Surprise = (0.75, 0.5)

    def map(labels, component: int, *, discrete: bool = False):
        continuous = np.array([getattr(EmotionMap, x).value[component] for x in labels])
        if not discrete:
            return continuous
        return np.array(continuous*4+4, dtype=np.int32)
    
    def map_both(labels, *, discrete: bool = False):
        return np.stack((
            EmotionMap.map(labels, 0, discrete=discrete),
            EmotionMap.map(labels, 1, discrete=discrete),
        ), axis=1)

    def reverse_map(valence, arousal, *, discrete: bool):
        point = np.array([valence, arousal])
        if discrete:
            point = (point-4)/4
        enum_values = np.array([emotion.value for emotion in EmotionMap])
        distances = np.linalg.norm(enum_values - point, axis=1)
        min_index = np.argmin(distances)
        closest_emotion = np.array([emotion.name for emotion in EmotionMap])[min_index]
    
        return closest_emotion

class TwoModelWrapper:
    def __init__(self, valenceModel, arousalModel):
        self.valenceModel = valenceModel
        self.arousalModel = arousalModel
        self._discrete = False

    def discrete(self):
        self._discrete = True
        return self

    def fit(self, X_train, y_train):
        self.valenceModel.fit(X_train, EmotionMap.map(y_train, 0, discrete=self._discrete))
        self.arousalModel.fit(X_train, EmotionMap.map(y_train, 1, discrete=self._discrete))
        return self
    
    def predict(self, X_test):
        valence = self.valenceModel.predict(X_test)
        arousal = self.arousalModel.predict(X_test)
        return np.array([EmotionMap.reverse_map(valence[i], arousal[i], discrete=self._discrete) for i in range(len(valence))])

    @property
    def best_params_(self):
        return [self.valenceModel.best_params_, self.arousalModel.best_params_]


class SklearnTwoModelWrapper(TwoModelWrapper):
    def __init__(self, baseModel):
        super().__init__(clone(baseModel), clone(baseModel))

class KerasWrapper:
    def __init__(self, model):
        self.model = model

    def compile(self, *args, **kwargs):
        return self.model.compile(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)
    
    def evaluate(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)
    
    def fit(self, X_train, Y_train, *args, validation_data, **kwargs):
        validation_data = (validation_data[0], EmotionMap.map_both(validation_data[1]))
        return self.model.fit(X_train, EmotionMap.map_both(Y_train), *args, validation_data=validation_data, **kwargs)