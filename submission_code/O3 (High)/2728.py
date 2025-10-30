from typing import List

class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        # Sort each row in decreasing order so that the k-th removal
        # from every row is located at index k
        for row in nums:
            row.sort(reverse=True)
        
        max_len = max(len(row) for row in nums)          # longest row length
        score   = 0
        
        # For every “round” (i.e. column after sorting)
        for col in range(max_len):
            best = 0
            for row in nums:
                if col < len(row):                       # row still has elements left
                    best = max(best, row[col])
            score += best                                # add the largest removed value
        return score