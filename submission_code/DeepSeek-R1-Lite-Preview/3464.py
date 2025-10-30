from typing import List

class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        # Compute prefix sums S[i] = sum_{k=0}^{i-1} nums[k] * (-1)^k
        S = [0] * (n + 1)
        for i in range(n):
            S[i+1] = S[i] + nums[i] * ((-1) ** i)
        
        # Total cost without any splits
        total_without_split = S[n]
        
        # Find the maximum cost with one split
        max_with_split = float('-inf')
        for j in range(n):
            sign = (-1) ** (j + 1)
            cost = S[j+1] + sign * (S[n] - S[j+1])
            if cost > max_with_split:
                max_with_split = cost
        
        # The answer is the maximum of total_without_split and max_with_split
        return max(total_without_split, max_with_split)