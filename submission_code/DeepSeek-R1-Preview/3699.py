from typing import List
from collections import defaultdict

class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        for q in range(n):
            for r in range(q + 2, n):
                # Compute left products: p ranges from 0 to q-2
                left = defaultdict(int)
                for p in range(q - 1):
                    pr = nums[p] * nums[r]
                    left[pr] += 1
                # Compute right products and add to count
                for s in range(r + 2, n):
                    qs = nums[q] * nums[s]
                    count += left.get(qs, 0)
        return count