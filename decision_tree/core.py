import numpy as np


def calc_gini_score(y) -> float:
    if len(y) == 0:
        return 0
    y_unique = np.unique(y)
    res = 0
    for value in y_unique:
        res += (np.count_nonzero(y == value) / len(y)) ** 2
    return 1 - res


def calc_gain(input_y, output_y_left, output_y_right) -> float:
    assert len(input_y) == (len(output_y_left) + len(output_y_right))
    input_gini_impurity = calc_gini_score(input_y)
    output_gini_impurity = calc_gini_score(output_y_left) * (
        len(output_y_left) / len(input_y)
    ) + calc_gini_score(output_y_right) * (len(output_y_right) / len(input_y))
    return input_gini_impurity - output_gini_impurity


def calc_best_split_feature(x, y, feature_mask=None):
    num_of_features = x.shape[1]
    max_gain = -1
    max_gain_feature_index = -1
    max_gain_threshold = -1
    for feature_index in range(num_of_features):
        if feature_mask is not None and feature_mask[feature_index]:
            continue
        feature_values = x[:, feature_index]
        feature_values_unique = np.unique(feature_values)
        for feature_value_threshold in feature_values_unique:
            y_left = y[feature_values <= feature_value_threshold]
            y_right = y[feature_values > feature_value_threshold]
            gain = calc_gain(y, y_left, y_right)
            if max_gain < gain:
                max_gain = gain
                max_gain_feature_index = feature_index
                max_gain_threshold = feature_value_threshold
    return max_gain, max_gain_feature_index, max_gain_threshold


class Node:
    def __init__(self, x, y, num_of_class, max_depth, current_depth, feature_mask=None):
        self.prob = [np.count_nonzero(y == i) / len(y) for i in range(num_of_class)]

        if current_depth <= max_depth:
            self.is_leaf = False

            (
                self.gain,
                self.split_feature_index,
                self.split_threshold,
            ) = calc_best_split_feature(x, y, feature_mask=feature_mask)

            feature_values = x[:, self.split_feature_index]
            x_left = x[feature_values <= self.split_threshold]
            x_right = x[feature_values > self.split_threshold]
            y_left = y[feature_values <= self.split_threshold]
            y_right = y[feature_values > self.split_threshold]

            if len(y_left) == 0 or len(y_right) == 0:
                self.is_leaf = True
            else:
                self.left_node = Node(
                    x_left, y_left, num_of_class, max_depth, current_depth + 1
                )
                self.right_node = Node(
                    x_right, y_right, num_of_class, max_depth, current_depth + 1
                )
        else:
            self.is_leaf = True

    def feature_importance(self, importance_value):
        if not self.is_leaf:
            importance_value[self.split_feature_index] += self.gain
            self.left_node.feature_importance(importance_value)
            self.right_node.feature_importance(importance_value)
        return importance_value

    def output(self, x):
        if self.is_leaf:
            return self.prob
        else:
            feature_value = x[self.split_feature_index]
            if feature_value <= self.split_threshold:
                return self.left_node.output(x)
            else:
                return self.right_node.output(x)


class Tree:
    def __init__(self, num_of_class, max_depth=5):
        self.num_of_class = num_of_class
        self.max_depth = max_depth
        self.root_node = None
        self.feature_importance = None

    def fit(self, x, y, feature_mask=None):
        self.root_node = Node(
            x, y, self.num_of_class, self.max_depth, 1, feature_mask=feature_mask
        )
        self.feature_importance = self.root_node.feature_importance(
            np.array([0.0] * x.shape[1])
        )

    def predict_proba(self, x):
        return [self.root_node.output(values) for values in x]

    def predict(self, x):
        return [np.argmax(self.root_node.output(values)) for values in x]
