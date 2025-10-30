from collections import defaultdict
from typing import List

class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        dp = defaultdict(int)
        max_length = 0
        for num in nums:
            for adjusted in [num, num + 1]:
                prev = adjusted - 1
                current = dp.get(prev, 0) + 1
                if current > dp.get(adjusted, 0):
                    dp[adjusted] = current
                if dp[adjusted] > max_length:
                    max_length = dp[adjusted]
        return max_length