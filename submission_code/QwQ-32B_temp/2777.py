from typing import List

class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        diff = []
        for i in range(n):
            prefix = nums[:i+1]
            suffix = nums[i+1:]
            p_dist = len(set(prefix))
            s_dist = len(set(suffix))
            diff.append(p_dist - s_dist)
        return diff