from typing import List

class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # Record the last occurrence (right boundary) of each number
        last_occurrence = {}
        for i, x in enumerate(nums):
            last_occurrence[x] = i
        
        # Traverse and count valid cut positions
        cuts = 0
        curr_max_r = 0
        # We only consider cuts after index i for i in [0..n-2]
        for i in range(n - 1):
            curr_max_r = max(curr_max_r, last_occurrence[nums[i]])
            # If no interval extends beyond i, we can cut here
            if curr_max_r == i:
                cuts += 1
        
        # Each valid cut position can either be chosen or not
        return pow(2, cuts, MOD)