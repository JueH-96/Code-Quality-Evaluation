from typing import List
import math

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        # transform: a[i] = nums[i] - i, so that condition becomes
        # for balanced subsequence indices i1 < i2 ...:
        #   nums[i_j] - nums[i_j-1] >= i_j - i_j-1
        # <=> (nums[i_j]-i_j) >= (nums[i_j-1]-i_j-1)
        # Hence, the "a" values should form a non-decreasing sequence.
        a = [nums[i] - i for i in range(n)]
        
        # Coordinate compression for the a array values.
        sorted_unique = sorted(set(a))
        comp = {val: idx for idx, val in enumerate(sorted_unique)}
        size = len(sorted_unique)
        
        # Fenwick Tree (Binary Indexed Tree) for range maximum queries.
        # We'll use 1-indexed internal tree.
        class Fenw:
            def __init__(self, n):
                self.n = n
                self.data = [-math.inf] * (n+1)
            def update(self, idx, value):
                # idx is 1-indexed.
                while idx <= self.n:
                    if value > self.data[idx]:
                        self.data[idx] = value
                    idx += idx & -idx
            def query(self, idx):
                # returns max in range [1, idx]
                ret = -math.inf
                while idx:
                    ret = max(ret, self.data[idx])
                    idx -= idx & -idx
                return ret
        
        fenw = Fenw(size)
        res = -math.inf
        
        # Process in order of indices
        for i in range(n):
            # compressed index for current a[i]
            idx = comp[a[i]]  # 0-indexed for compression
            # We want to get maximum DP sum among all indices j with transformed value <= a[i]
            bestPrev = fenw.query(idx+1)  # fenw tree is built 1-indexed; now idx+1 covers all <= a[i]
            # For starting a new subsequence with just the current element,
            # we can also consider 0 as the dp of an empty subsequence.
            current_dp = max(bestPrev, 0) + nums[i]
            # Update the fenw tree at position idx+1 with current_dp.
            fenw.update(idx+1, current_dp)
            res = max(res, current_dp)
        
        return res