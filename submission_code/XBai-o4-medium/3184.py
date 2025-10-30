from bisect import bisect_left
from math import inf
from typing import List

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [-inf] * (self.n + 1)  # 1-based indexing
    
    def update(self, idx, value):
        while idx <= self.n:
            if value > self.tree[idx]:
                self.tree[idx] = value
                idx += idx & -idx
            else:
                break
    
    def query(self, idx):
        res = -inf
        while idx > 0:
            if self.tree[idx] > res:
                res = self.tree[idx]
            idx -= idx & -idx
        return res

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        # Generate all a_i = nums[i] - i
        all_a = [nums[i] - i for i in range(n)]
        # Coordinate compression
        sorted_unique = sorted({a for a in all_a})
        m = len(sorted_unique)
        # Create Fenwick Tree
        fenwick = FenwickTree(m)
        max_sum = -inf
        for i in range(n):
            a_i = all_a[i]
            # Find rank in sorted_unique
            rank = bisect_left(sorted_unique, a_i)
            fenwick_idx = rank + 1  # 1-based
            # Query the maximum up to this rank
            prev_max = fenwick.query(fenwick_idx)
            if prev_max == -inf:
                current_sum = nums[i]
            else:
                current_sum = max(prev_max, 0) + nums[i]
            if current_sum > max_sum:
                max_sum = current_sum
            # Update the Fenwick Tree
            fenwick.update(fenwick_idx, current_sum)
        return max_sum