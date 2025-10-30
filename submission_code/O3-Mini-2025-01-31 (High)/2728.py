from typing import List

class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        # Sort each row in descending order.
        # This allows us to simulate the removal of the largest element
        # in each round by simply iterating over each row index.
        for row in nums:
            row.sort(reverse=True)
            
        score = 0
        # Determine the number of rounds required.
        # The matrix becomes empty when all rows have been exhausted.
        max_rounds = max(len(row) for row in nums)
        
        # For each round, select the r-th element from each row (if available)
        # which represents the largest element left in that row.
        # Then, add the maximum among these selections to the score.
        for i in range(max_rounds):
            candidates = [row[i] for row in nums if i < len(row)]
            score += max(candidates)
        
        return score