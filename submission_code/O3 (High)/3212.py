from typing import List

MOD = 10**9 + 7

class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        # Last position where each value appears
        last_pos = {}
        for i, val in enumerate(nums):
            last_pos[val] = i
        
        n = len(nums)
        cuts = 0          # how many places we are free to cut
        right = 0         # furthest last-occurrence seen so far
        
        for i, val in enumerate(nums):
            right = max(right, last_pos[val])   # interval we must keep together
            # if every interval that started so far ends here,
            # we are allowed to make a cut AFTER position i
            if i == right and i < n - 1:        # cannot cut after the last element
                cuts += 1
        
        # each independent cut position can be chosen or skipped
        return pow(2, cuts, MOD)