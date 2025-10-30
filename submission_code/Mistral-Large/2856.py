from typing import List
from collections import Counter

class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        total_distinct = len(set(nums))
        count = 0
        n = len(nums)
        window_counts = Counter()
        left = 0

        for right in range(n):
            window_counts[nums[right]] += 1

            while len(window_counts) == total_distinct:
                count += n - right
                window_counts[nums[left]] -= 1
                if window_counts[nums[left]] == 0:
                    del window_counts[nums[left]]
                left += 1

        return count