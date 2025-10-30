from typing import List
from collections import defaultdict
import math

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        count = defaultdict(int)
        for num in nums:
            count[num] += 1

        nums = sorted(count.keys())
        dp = defaultdict(int)
        dp[nums[0]] = count[nums[0]]

        for i in range(1, len(nums)):
            dp[nums[i]] = max(dp[nums[i]], count[nums[i]])
            root = int(math.sqrt(nums[i]))
            if root * root == nums[i] and root in dp:
                dp[nums[i]] = max(dp[nums[i]], dp[root] + count[nums[i]])

        return max(dp.values())