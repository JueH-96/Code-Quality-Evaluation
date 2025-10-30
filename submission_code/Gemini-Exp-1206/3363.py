from typing import List
from collections import defaultdict

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        n = len(nums)
        ans = []
        counts = defaultdict(int)
        for i in range(n):
            counts[nums[i]] += freq[i]
            max_freq = 0
            for count in counts.values():
                if count > max_freq:
                    max_freq = count
            ans.append(max_freq if any(counts.values()) else 0)
        return ans