from typing import List

class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        # Sort each row in non-decreasing order.
        for row in nums:
            row.sort()
        
        score = 0
        # Number of operations is equal to the maximum row length.
        max_cols = max(len(row) for row in nums)
        
        # For each operation, we pick the ith largest element from rows that have at least i elements.
        for i in range(1, max_cols + 1):
            current_max = 0
            for row in nums:
                if len(row) >= i:
                    # row[-i] gives the ith largest element in the row.
                    current_max = max(current_max, row[-i])
            score += current_max
        
        return score