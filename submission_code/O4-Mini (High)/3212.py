from typing import List

class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        # Record the last occurrence index of each number
        last = {}
        for i, x in enumerate(nums):
            last[x] = i
        
        # Count the positions where we can cut such that
        # no number spans across the cut.
        cuts = 0
        max_last = 0
        # We only consider cuts between i and i+1 for i in [0..n-2]
        for i in range(n - 1):
            x = nums[i]
            max_last = max(max_last, last[x])
            # If the farthest last occurrence among seen numbers is i,
            # then no number crosses this boundary, so we can cut here.
            if max_last == i:
                cuts += 1
        
        # Each valid cut position can be chosen or not, so 2^cuts possibilities
        return pow(2, cuts, MOD)