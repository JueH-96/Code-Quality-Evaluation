from typing import List
from collections import Counter

class Solution:
    def isGood(self, nums: List[int]) -> bool:
        max_num = max(nums)
        if len(nums) != max_num + 1:
            return False
        count = Counter(nums)
        if count[max_num] != 2:
            return False
        for i in range(1, max_num):
            if count[i] != 1:
                return False
        return True