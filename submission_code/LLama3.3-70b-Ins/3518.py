from typing import List
import itertools

class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        max_score = float('-inf')
        
        # Generate all possible combinations of 4 indices from the array b
        for indices in itertools.combinations(range(len(b)), 4):
            # Check if the indices are in ascending order
            if indices[0] < indices[1] < indices[2] < indices[3]:
                # Calculate the score for the current combination of indices
                score = a[0] * b[indices[0]] + a[1] * b[indices[1]] + a[2] * b[indices[2]] + a[3] * b[indices[3]]
                # Update the maximum score if the current score is higher
                max_score = max(max_score, score)
        
        return max_score