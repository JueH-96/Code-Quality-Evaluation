from typing import List
from collections import defaultdict

class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        max_len = 0
        freq = defaultdict(int)
        left = 0
        for right, num in enumerate(nums):
            freq[num] += 1
            while any(count > k for count in freq.values()):
                freq[nums[left]] -= 1
                left += 1
            max_len = max(max_len, right - left + 1)
        return max_len