from typing import List

class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        # The first operation's score is fixed by the first two elements.
        target = nums[0] + nums[1]
        count = 1
        i = 2
        n = len(nums)
        
        # Continue pairing subsequent elements in order as long as
        # their sum matches the initial target.
        while i + 1 < n and nums[i] + nums[i + 1] == target:
            count += 1
            i += 2
        
        return count