from typing import List
from collections import defaultdict

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        n = len(nums)
        id_counts = defaultdict(int)
        max_count = 0
        ans = [0]*n

        for i in range(n):
            id_counts[nums[i]] += freq[i]
            max_count = max(max_count, id_counts[nums[i]])
            ans[i] = max_count

        return ans