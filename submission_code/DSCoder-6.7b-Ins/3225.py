from collections import Counter
from typing import List

class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        count = Counter()
        left = 0
        ans = 0
        for right, num in enumerate(nums):
            count[num] += 1
            while count[num] > k:
                count[nums[left]] -= 1
                left += 1
            ans = max(ans, right - left + 1)
        return ans