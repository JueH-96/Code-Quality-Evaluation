from typing import List
import bisect
import math

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        s = [nums[i] - i for i in range(n)]
        sorted_unique_s = sorted(set(s))
        m = len(sorted_unique_s)
        
        class FenwickTree:
            def __init__(self, size):
                self.N = size + 1  # 1-based indexing
                self.tree = [-math.inf] * self.N
            def update(self, index, value):
                while index < self.N:
                    if value > self.tree[index]:
                        self.tree[index] = value
                    else:
                        break
                    index += index & -index
            def query(self, index):
                res = -math.inf
                while index > 0:
                    if self.tree[index] > res:
                        res = self.tree[index]
                    index -= index & -index
                return res
        
        fenwick = FenwickTree(m)
        overall_max = -math.inf
        for i in range(n):
            s_i = s[i]
            rank = bisect.bisect_left(sorted_unique_s, s_i) + 1
            current_max = fenwick.query(rank - 1)
            dp_i = max(nums[i] + current_max, nums[i])
            fenwick.update(rank, dp_i)
            if dp_i > overall_max:
                overall_max = dp_i
        return overall_max