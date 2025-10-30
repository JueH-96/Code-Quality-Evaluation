from typing import List

class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        # If there are fewer than 2 elements no operation is possible.
        if len(nums) < 2:
            return 0
        
        # The score of the first operation fixes the target score.
        target = nums[0] + nums[1]
        count = 0
        i = 0
        
        # As long as we have at least two elements and their sum equals the target,
        # we can perform another operation.
        while i + 1 < len(nums) and nums[i] + nums[i + 1] == target:
            count += 1
            i += 2
        
        return count