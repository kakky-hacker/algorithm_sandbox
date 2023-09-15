from unittest import TestCase

import numpy as np
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
