from collections import defaultdict
from typing import List

class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        dp = defaultdict(int)
        max_len = 0
        for x in nums:
            temp_x = dp[x - 1] + 1
            temp_x_plus_1 = dp[x] + 1
            dp[x] = max(dp[x], temp_x)
            dp[x + 1] = max(dp[x + 1], temp_x_plus_1)
            current_max = max(dp[x], dp[x + 1])
            if current_max > max_len:
                max_len = current_max
        return max_len