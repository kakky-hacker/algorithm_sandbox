import warnings

warnings.simplefilter("ignore")

from unittest import TestCase

import numpy as np
import pandas as pd
from core import *
from random_forest import *


class test_calc_gini_score(TestCase):
    def test1(self):
        y = [0, 0, 0, 0]
        assert calc_gini_score(y) == 0

    def test2(self):
        y = [1, 1, 1, 1]
        assert calc_gini_score(y) == 0

    def test3(self):
        y = [0, 1, 0, 1]
        assert calc_gini_score(y) == 0.5

    def test3(self):
        y = [0, 1, 2, 3]
        assert calc_gini_score(y) == 0.75

    def test4(self):
        y = []
        assert calc_gini_score(y) == 0

    def test5(self):
        y = [1]
        assert calc_gini_score(y) == 0


class test_calc_gain(TestCase):
    def test1(self):
        input_y = [0, 1, 0, 1]
        output_y_left = [0, 0]
        output_y_right = [1, 1]
        assert calc_gain(input_y, output_y_left, output_y_right) == 0.5

    def test2(self):
        input_y = [0, 1, 0, 1]
        output_y_left = [0, 1]
        output_y_right = [1, 0]
        assert calc_gain(input_y, output_y_left, output_y_right) == 0.0

    def test3(self):
        input_y = [1, 1, 1, 1]
        output_y_left = [1, 1]
        output_y_right = [1, 1]
        assert calc_gain(input_y, output_y_left, output_y_right) == 0.0

    def test4(self):
        input_y = [0, 0, 1, 1, 2, 2, 3, 3]
        output_y_left = [0, 0, 2, 2]
        output_y_right = [1, 1, 3, 3]
        assert calc_gain(input_y, output_y_left, output_y_right) == 0.25


class test_calc_best_split_feature(TestCase):
    def test1(self):
        x = np.array([[0, 1], [0, 2], [1, 3], [1, 4]])
        y = np.array([0, 0, 1, 1])
        assert calc_best_split_feature(x, y) == (0.5, 0, 0)

    def test2(self):
        x = np.array([[0, 1], [0, 2], [0, 3], [1, 4]])
        y = np.array([0, 0, 1, 1])
        assert calc_best_split_feature(x, y) == (0.5, 1, 2)


class test_Node(TestCase):
    def test1(self):
        x = np.array([[0, 1], [0, 2], [0, 3], [1, 4]])
        y = np.array([0, 0, 1, 1])
        node = Node(x, y, num_of_class=2, max_depth=1, current_depth=1)
        assert node.output([0, 2]) == [1.0, 0.0]
        assert node.output([1, 4]) == [0.0, 1.0]


class test_Tree(TestCase):
    def test1(self):
        x = np.array([[0, 1], [0, 2], [0, 3], [1, 4]])
        y = np.array([0, 0, 1, 1])
        tree = Tree(num_of_class=2, max_depth=3)
        tree.fit(x, y)
        assert tree.predict_proba([[0, 2]]) == [[1.0, 0.0]]
        assert tree.predict_proba([[1, 4]]) == [[0.0, 1.0]]
        assert all(tree.feature_importance == [0.0, 0.5])


class test_create_mask(TestCase):
    def test1(self):
        mask = create_mask(mask_length=10, max_num_of_zero_values=4)
        assert len(mask) == 10
        assert sum(mask) == 6


class test_create_shadow_features(TestCase):
    def test1(self):
        features = np.array(
            [[3, 5, 7, 11], [6, 10, 14, 22], [9, 15, 21, 33]], dtype=np.float32
        )
        shadow_features = create_shadow_features(features=features)
        assert features.shape == shadow_features.shape
        assert all(features.sum(axis=0) == shadow_features.sum(axis=0))


class test_calc_hit_feature(TestCase):
    def test1(self):
        original_feature_importance = np.array([2, 0, 3, 8, 1])
        shadow_feature_importance = np.array([1, 0, 1, 0, 0])
        hit_feature = calc_hit_feature(
            original_feature_importance, shadow_feature_importance
        )
        assert all(hit_feature == np.array([1, 0, 1, 1, 0]))


class test_calc_num_of_hit_per_feature(TestCase):
    def test1(self):
        original_feature_importances = np.array([[2, 0, 3, 8, 1], [4, 1, 5, 7, 2]])
        shadow_feature_importances = np.array([[1, 0, 1, 0, 0], [1, 0, 2, 1, 0]])
        num_of_hit_per_feature = calc_num_of_hit_per_feature(
            original_feature_importances, shadow_feature_importances
        )
        assert all(num_of_hit_per_feature == np.array([2, 0, 2, 2, 0]))


class test_calc_ttest_bin(TestCase):
    def test1(self):
        assert 0.184 < calc_ttest_bin(0.5, 100, 55) < 0.185


class test_RandomForest(TestCase):
    def test_fit(self):
        train = pd.read_csv("data/titanic/train.csv")

        train["Age"] = train["Age"].fillna(train["Age"].median())
        train["Embarked"] = train["Embarked"].fillna("S")
        train.loc[train["Sex"] == "male", "Sex"] = 0
        train.loc[train["Sex"] == "female", "Sex"] = 1
        train.loc[train["Embarked"] == "S", "Embarked"] = 0
        train.loc[train["Embarked"] == "C", "Embarked"] = 1
        train.loc[train["Embarked"] == "Q", "Embarked"] = 2

        train_x = train[
            ["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]
        ].values[: int(len(train) * 0.8)]
        train_y = train["Survived"].values[: int(len(train) * 0.8)]
        test_x = train[
            ["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]
        ].values[int(len(train) * 0.8) :]
        test_y = train["Survived"].values[int(len(train) * 0.8) :]

        model = RandomForest(
            num_of_class=2, max_features=4, max_depth=5, n_estimators=5
        )
        model.fit(train_x, train_y)

        pred_y = model.predict(test_x)
        accuracy = sum([a == b for a, b in zip(pred_y, test_y)]) / len(test_y)
        print(pred_y, accuracy)
        assert accuracy > 0.86
