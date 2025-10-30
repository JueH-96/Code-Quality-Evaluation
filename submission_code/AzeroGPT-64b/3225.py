from collections import defaultdict
from typing import List


class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        map = defaultdict(int)
        left, right, res = 0, 0, 0
        while right < len(nums):
            map[nums[right]] += 1
            if max(map.values()) > k:
                while left <= right and max(map.values()) > k:
                    map[nums[left]] -= 1
                    left += 1
            res = max(res, right - left + 1)
            right += 1
        return res