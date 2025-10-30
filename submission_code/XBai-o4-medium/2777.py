from typing import List

class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        diff = []
        for i in range(n):
            prefix = nums[:i+1]
            suffix = nums[i+1:]
            count_p = len(set(prefix))
            count_s = len(set(suffix))
            diff.append(count_p - count_s)
        return diff