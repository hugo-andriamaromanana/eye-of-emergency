from collections import Counter
from typing import List, Tuple
from numpy import random, swapaxes
from numpy._typing import NDArray
from tree_models.decision_tree import CustomDecisionTree

class CustomRandomForest:
    '''
    '''
    def __init__(self, number_trees=25, minimum_samples_split=2, maximum_depth=5):
        self.number_trees = number_trees
        self.minimum_samples_split = minimum_samples_split
        self.maximum_depth = maximum_depth
        self.decision_trees = []
        
    @staticmethod
    def _sample(dataframe: NDArray, targets: NDArray) -> Tuple[NDArray, NDArray]:
        '''
        Function to create a random saple of the dataframe
        
        Input:        
            dataframe, NDArray : The matrix of the values to predict
            targets, NDArray : The matrix of target values
        Output:
            Tuple[Dataframe, NDArray] : The tuple of the sampled dataframe and the sampled targets
        '''
        number_rows, _ = dataframe.shape
        samples = random.choice(a=number_rows, size=number_rows, replace=True)
        return dataframe[samples], targets[samples]
        
    def fit(self, dataframe: NDArray, targets: NDArray) -> None:
        '''
        Function to build and fit the random forest to a datafrale and its target values 
        
        Input:        
            dataframe, NDarray : The matrix of the values to predict
            targets, NDArray : The matrix of target values
        '''
        if len(self.decision_trees) > 0:
            self.decision_trees = []
            
        number_built = 0
        while number_built < self.number_trees:
            try:
                decision_tree = CustomDecisionTree(
                    minimum_samples_split=self.minimum_samples_split,
                    maximum_depth=self.maximum_depth
                )
                _sampled_dataframe, _sampled_targets = self._sample(dataframe, targets)
                decision_tree.fit(_sampled_dataframe, _sampled_targets)
                self.decision_trees.append(decision_tree)
                number_built += 1
            except Exception as e:
                continue
    
    def predict(self, dataframe: NDArray) -> List[int]:
        '''
        Function to predict targets corresponding to values
        
        Input:        
            dataframe, NDArray : The matrix of the values to predict
        Output:
            List[int] : The list of predicted target values
        '''
        targets = []
        for tree in self.decision_trees:
            targets.append(tree.predict(dataframe))
        
        targets = swapaxes(a=targets, axis1=0, axis2=1)
        
        predictions = []
        for preds in targets:
            counter = Counter(preds)
            predictions.append(counter.most_common(1)[0][0])
        return predictions