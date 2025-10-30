from typing import List

class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        # Initialize prefix_min where prefix_min[t][j] is the minimal value of nums for type t up to j rotations
        prefix_min = [[0] * n for _ in range(n)]
        
        for t in range(n):
            curr_min = float('inf')
            for j in range(n):
                # Calculate the original position i for type t at rotation j
                i = (t - j) % n
                curr_val = nums[i]
                if curr_val < curr_min:
                    curr_min = curr_val
                prefix_min[t][j] = curr_min
        
        res = float('inf')
        # Try each possible number of rotations (k) up to n-1
        for k in range(n):
            total = 0
            for t in range(n):
                total += prefix_min[t][k]
            total += x * k
            if total < res:
                res = total
        
        return res