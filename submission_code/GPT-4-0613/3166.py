from typing import List
from collections import Counter

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        count = Counter(nums)
        return max(len(nums) // 2, max(count.values()))