from random import randint

import numpy as np
from core import Tree


def create_mask(mask_length, max_num_of_zero_values):
    mask = [False] * mask_length
    while max_num_of_zero_values < (mask_length - sum(mask)):
        mask[randint(0, mask_length - 1)] = True
    return mask


def create_shadow_features(features):
    rng = np.random.default_rng()
    len_of_features, num_of_features = features.shape
    shadow_x = [[] for _ in range(len_of_features)]
    for i in range(num_of_features):
        shadow_x = np.concatenate(
            [shadow_x, rng.permutation(features[:, i]).reshape((len_of_features, 1))],
            axis=1,
        )
    return shadow_x


def calc_hit_feature(original_feature_importance, shadow_feature_importance):
    return original_feature_importance > max(shadow_feature_importance)


def calc_num_of_hit_per_feature(
    original_feature_importances, shadow_feature_importances
):
    return np.sum(
        [
            calc_hit_feature(original_feature_importance, shadow_feature_importance)
            for original_feature_importance, shadow_feature_importance in zip(
                original_feature_importances, shadow_feature_importances
            )
        ],
        axis=0,
    )


class RandomForest:
    def __init__(self, num_of_class, max_features, n_estimators=100, max_depth=5):
        self.num_of_class = num_of_class
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, x, y):
        num_of_features = x.shape[1]

        # create trees
        self.trees = [
            Tree(self.num_of_class, max_depth=self.max_depth)
            for _ in range(self.n_estimators)
        ]

        # train trees
        for tree in self.trees:
            tree.fit(x, y, feature_mask=create_mask(num_of_features, self.max_features))

        # calc feature importance
        self.feature_importance = np.average(
            [tree.feature_importance for tree in self.trees], axis=0
        )

    def fit_with_boruta(self, x, y):
        num_of_features = x.shape[1] * 2
        feature_importances = np.array(
            [[] for _ in range(num_of_features)]
        )  # shape(num_of_features, 100)
        for _ in range(100):
            self.fit(np.concatenate([x, create_shadow_features(x)], axis=1), y)
            feature_importances = np.concatenate(
                [
                    feature_importances,
                    self.feature_importance.reshape(num_of_features, 1),
                ],
                axis=1,
            )
        hit_per_feature = calc_num_of_hit_per_feature(
            feature_importances[: len(feature_importances) // 2],
            feature_importances[len(feature_importances) // 2 :],
        )  # shape(num_of_features, 1)

    def predict(self, x):
        return np.argmax([tree.predict(x) for tree in self.trees], axis=1)
