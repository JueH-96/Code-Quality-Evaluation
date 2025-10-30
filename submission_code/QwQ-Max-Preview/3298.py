from typing import List
from collections import defaultdict

class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        dp = defaultdict(int)
        for a in nums:
            temp = defaultdict(int)
            for current_val in [a, a + 1]:
                prev_val = current_val - 1
                current_length = dp.get(prev_val, 0) + 1
                if current_length > temp[current_val]:
                    temp[current_val] = current_length
            for val in temp:
                if temp[val] > dp.get(val, 0):
                    dp[val] = temp[val]
        return max(dp.values(), default=0)