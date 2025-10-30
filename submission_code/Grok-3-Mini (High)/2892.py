from typing import List
import collections

class Solution:
    def isGood(self, nums: List[int]) -> bool:
        max_val = max(nums)
        if len(nums) != max_val + 1:
            return False
        counter = collections.Counter(nums)
        for i in range(1, max_val):
            if counter[i] != 1:
                return False
        if counter[max_val] != 2:
            return False
        return True