from typing import List

class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        # determine the longest sequential prefix
        # the prefix always starts with nums[0]
        n = len(nums)
        prefix = [nums[0]]
        for i in range(1, n):
            if nums[i] == prefix[-1] + 1:
                prefix.append(nums[i])
            else:
                break
        
        target = sum(prefix)
        # convert nums to a set for efficient membership checking
        nums_set = set(nums)
        
        # find smallest integer x >= target that is not in nums
        x = target
        while x in nums_set:
            x += 1
        return x