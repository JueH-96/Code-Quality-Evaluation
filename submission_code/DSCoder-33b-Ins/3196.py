from typing import List
from collections import Counter

class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        i = 0
        j = 0
        n = len(nums)
        counter = Counter()
        max_freq = 0
        while j < n:
            counter[nums[j]] += 1
            max_freq = max(max_freq, counter[nums[j]])
            if j - i + 1 > max_freq + k:
                counter[nums[i]] -= 1
                i += 1
            j += 1
        return max_freq