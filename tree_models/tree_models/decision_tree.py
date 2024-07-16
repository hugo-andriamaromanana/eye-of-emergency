from dataclasses import dataclass
from typing import Optional
from numpy import argmax, argwhere, array, bincount, log2, ndarray, random, unique, sum


class Node:

    def __init__(
        self,
        feature: int = 0,
        threshold: float = 0,
        left=None,
        right=None,
        *,
        value: Optional[float] = None
    ) -> None:
        pass
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right


class CustomDecisionTree:

    def __init__(self, maximum_depth=100, minimum_samples_split=2) -> None:
        self.maximum_depth: int = maximum_depth
        self.minimum_sample_split: int = minimum_samples_split
        self.root: Optional[Node] = None
        self.number_samples: int = 0
        self.number_features: int = 0
        self.number_class_labels: int = 0

    def _is_finished(self, depth: float) -> bool:
        """
        Boolean function if the walkthrough is finished (in optic to return the most common label)

        Input:
            depth, float : the defined depth into the tree walkthrough
        Output:
            bool : is the walkthrough finished
        Conditions:
            maximum_depth exedeed, minimum_sample_split insufficient or unique class label
        """
        return (
            depth > self.maximum_depth
            or self.number_samples < self.minimum_sample_split
            or self.number_class_labels == 1
        )

    def _entropy(self, target_values: ndarray) -> float:
        """
        Function to calculate the entropy, the average level of uncertainty. It is a great indicator of the node's potential and necessary to calculate information gain.

        Input:
            target_values, ndarray: The matrix of the target labels
        Output:
            float: the enthopy of the target values, between 0 and 1
        Mathematics expression:
            Sum(i -> n)P(xi)*logp(xi)
        """
        proportions = bincount(target_values) / len(target_values)
        return -sum(
            [
                proportion * log2(proportion)
                for proportion in proportions
                if proportion > 0
            ]
        )

    def _create_split(self, dataframe: ndarray, threshold: float) -> tuple:
        """
        Function to divide in two splits the dataframe indexes by a specific threshold.

        Input :
            dataframe, ndarray: The matrix of the values of the dataframe
            threshold, float: the number chosen to split in two the dataframe
        Output:
            tuple[ndarray, ndarray] : the two matrix of indexes of the split dataframe
        """
        left_indexes = argwhere(dataframe <= threshold).flatten()
        right_indexes = argwhere(dataframe > threshold).flatten()
        return left_indexes, right_indexes

    def _information_gain(
        self, dataframe: ndarray, target_values: ndarray, threshold: float
    ) -> float:
        """
        Function to calculate information gain, the substraction of the children enthropy to the parent (with correct proportion to keep a value between 0 and 1).

        Input:
            dataframe, ndarray: The matrix of the values of the dataframe
            target_values, ndarray: The matrix of the target labels
            threshold, float: the number chosen to split in two the dataframe
        Output:
            float: the information gain of the parent with the two children
        Mathematics expression:
            E(parent) - ( E(left_child) * (length_left_child / length_parent) + E(right_child) * (length_right_child / length_parent) )
        """
        parent_loss = self._entropy(target_values)
        left_indexes, right_indexes = self._create_split(dataframe, threshold)
        total_length, left_length, right_length = (
            len(target_values),
            len(left_indexes),
            len(right_indexes),
        )

        if left_length == 0 or right_length == 0:
            return 0

        child_loss = left_length / total_length * self._entropy(
            target_values[left_indexes]
        ) + right_length / total_length * self._entropy(target_values[right_indexes])
        return parent_loss - child_loss

    def _best_split(
        self, dataframe: ndarray, target_values: ndarray, features: ndarray
    ) -> tuple[int, float]:
        """
        Function to find the best split with specific feature and threshold.

        Input:
            dataframe, ndarray: The matrix of the values of the dataframe
            target_values, ndarray: The matrix of the target labels
            features, float: the matrix of  indexes of features
        Output:
            tuple[int, float]: The best feature index and the best threshold
        """
        split = {"score": -1, "feature": None, "threshold": None}

        for feature in features:
            dataframe_by_feature = dataframe[:, feature]
            thresholds = unique(dataframe_by_feature)
            for threshold in thresholds:
                score = self._information_gain(
                    dataframe_by_feature, target_values, threshold
                )

                if score > split["score"]:
                    split["score"] = score
                    split["feature"] = feature
                    split["threshold"] = threshold

        return split["feature"], split["threshold"]

    def _most_common_label(self, target_values: ndarray) -> float:
        """
        Fuction to find the most common label in a serie.

        Input:
            target_values, ndarray: The matrix of the target labels
        Output:
            float: the label the most present in the target values
        """
        return float(argmax(bincount(target_values)))

    def _build_tree(
        self, dataframe: ndarray, target_values: ndarray, depth: float = 0
    ) -> Node:
        """
        Function to build the decision tree recursively to a maximum depth.

        Input:
            dataframe, ndarray: The matrix of the values of the dataframe
            target_values, ndarray: The matrix of the target labels
            depth, float = 0: the depth into the tree walk-through
        Output:
            Node: the node of the actual depth with the best feature, threshold and the two split children
        """
        self.number_samples, self.number_features = dataframe.shape
        self.number_class_labels = len(unique(target_values))

        if self._is_finished(depth):
            return Node(value=self._most_common_label(target_values))

        random_features = random.choice(
            self.number_features, self.number_features, replace=False
        )
        best_feature, best_threshold = self._best_split(
            dataframe, target_values, random_features
        )

        left_indexes, right_indexes = self._create_split(
            dataframe[:, best_feature], best_threshold
        )
        left_child, right_child = self._build_tree(
            dataframe[left_indexes, :], target_values[left_indexes], depth+1
        ), self._build_tree(
            dataframe[right_indexes, :], target_values[right_indexes], depth+1
        )
        return Node(best_feature, best_threshold, left_child, right_child)

    def _traverse_tree(self, serie: ndarray, node: Optional[Node]) -> float:
        """
        Function to find the best value by traversing recursively the tree.

        Input:
            serie, ndarray: The matrix of the value of a dataframe's observation serie
            node, Optional[Node]: the actual node into the walk-through
        Output:
            float: the value of the best leaf
        """
        if not node:
            raise Exception("The model need to have been train before predictions")
        if node.value:  # The nodes with values are the leafs of the tree
            return node.value

        if serie[node.feature] <= node.threshold:
            return self._traverse_tree(serie, node.left)
        return self._traverse_tree(serie, node.right)

    def fit(self, dataframe: ndarray, target_values: ndarray) -> None:
        """
        Function to build a tree with a dataframe and the target_values corresponding.

        Input:
            dataframe, ndarray: The matrix of the values of the dataframe
            target_values, ndarray: The matrix of the target labels
        Output:
            None
        Self output:
            self.root, Node : self._build_tree method
        """
        self.root = self._build_tree(dataframe, target_values)

    def predict(self, dataframe: ndarray) -> ndarray:
        """
        Function to predict target values from a dataframe.

        Prerequisite:
            Fit training dataframe before
        Input:
            dataframe, ndarray: The matrix of the values of the dataframe
        Output:
            ndarray: The matrix of the predicted target labels
        """
        return array(self._traverse_tree(serie, self.root) for serie in dataframe)
