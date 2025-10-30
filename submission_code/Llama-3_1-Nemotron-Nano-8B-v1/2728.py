from collections import deque
from typing import List

class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        deques = [deque(sorted(row, reverse=True)) for row in nums]
        score = 0
        
        while True:
            non_empty = [d for d in deques if d]
            if not non_empty:
                break
            current_maxes = [d[0] for d in non_empty]
            max_val = max(current_maxes)
            score += max_val
            for d in non_empty:
                d.popleft()
        
        return score