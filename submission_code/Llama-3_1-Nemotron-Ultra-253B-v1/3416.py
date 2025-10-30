from collections import defaultdict
from typing import List

class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        if not nums:
            return 0
        k = len(str(nums[0]))
        counters = [defaultdict(int) for _ in range(k)]
        n = len(nums)
        for num in nums:
            s = str(num)
            for i in range(k):
                counters[i][s[i]] += 1
        total = 0
        for counter in counters:
            total_pairs = n * (n - 1) // 2
            same_pairs = sum(c * (c - 1) // 2 for c in counter.values())
            total += total_pairs - same_pairs
        return total