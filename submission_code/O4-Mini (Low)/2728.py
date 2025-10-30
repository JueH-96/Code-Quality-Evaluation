from typing import List

class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        # Sort each row in non-decreasing order
        for row in nums:
            row.sort()
        
        score = 0
        # Determine the maximum row length to know how many rounds we'll have
        max_len = max(len(row) for row in nums)
        
        # In each round k, we remove the k-th largest element from each row (if it exists)
        for k in range(1, max_len + 1):
            # Track the largest removed in this round
            round_max = 0
            for row in nums:
                if len(row) >= k:
                    # The k-th largest is at index -k
                    val = row[-k]
                    if val > round_max:
                        round_max = val
            score += round_max
        
        return score