from typing import List
import bisect
import sys
sys.setrecursionlimit(1 << 25)

class FenwickTree:
    def __init__(self, size):
        self.N = size + 2
        self.tree = [-float('inf')] * (self.N)
    
    def update(self, index, value):
        while index < self.N:
            if value > self.tree[index]:
                self.tree[index] = value
            index += index & -index
    
    def query(self, index):
        result = -float('inf')
        while index > 0:
            if self.tree[index] > result:
                result = self.tree[index]
            index -= index & -index
        return result

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        a = [nums[j] - j for j in range(n)]
        sorted_unique_a = sorted(set(a))
        m = len(sorted_unique_a)
        fenwick = FenwickTree(m)
        dp = [0] * n
        for i in range(n):
            rank_a_i = bisect.bisect_right(sorted_unique_a, a[i])
            if rank_a_i > 0:
                query_result = fenwick.query(rank_a_i - 1)
            else:
                query_result = -float('inf')
            dp[i] = max(nums[i], query_result + nums[i])
            rank_a_i_compressed = bisect.bisect_right(sorted_unique_a, a[i])
            fenwick.update(rank_a_i_compressed, dp[i])
        return max(dp)