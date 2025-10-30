from typing import List
import itertools

class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)
        max_score = float('-inf')
        
        # Generate all combinations of 4 indices from b
        for indices in itertools.combinations(range(n), 4):
            score = a[0] * b[indices[0]] + a[1] * b[indices[1]] + a[2] * b[indices[2]] + a[3] * b[indices[3]]
            max_score = max(max_score, score)
        
        return max_score