from typing import List

class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        # Sort each row so we can pick off the largest remaining elements by index
        for row in nums:
            row.sort()
        
        score = 0
        # The number of rounds is the length of the longest row
        max_rounds = max(len(row) for row in nums)
        
        # In each round t, each row that has at least t elements contributes its
        # t-th largest element (i.e. row[-t]) and we add the maximum of those to the score.
        for t in range(1, max_rounds + 1):
            current_max = 0
            for row in nums:
                if len(row) >= t:
                    current_max = max(current_max, row[-t])
            score += current_max
        
        return score