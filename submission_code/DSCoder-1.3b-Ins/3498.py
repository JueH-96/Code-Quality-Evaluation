from collections import defaultdict
from typing import List

class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if n % 2 != 0:
            return -1

        nums.sort()
        min_changes = 0
        freq = defaultdict(int)

        for i in range(n):
            freq[nums[i]] += 1

        for i in range(k):
            freq_i = defaultdict(int)
            for j in range(i, n, k):
                freq_i[nums[j]] += 1
            min_changes += sum((abs(v_i - v_j) ** 2 for v_i, v_j in freq_i.items()))

        return min_changes