import bisect
from typing import List

class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [-float('inf')] * (self.size + 1)  # 1-based indexing

    def update(self, index, value):
        while index <= self.size:
            if value > self.tree[index]:
                self.tree[index] = value
            else:
                break  # No need to proceed if the new value is smaller
            index += index & -index

    def query(self, index):
        res = -float('inf')
        while index > 0:
            if self.tree[index] > res:
                res = self.tree[index]
            index -= index & -index
        return res

    def get(self, index):
        return self.tree[index]

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        a_list = [nums[i] - i for i in range(n)]
        sorted_a = sorted(a_list)
        ft_size = n + 1
        fenwick = FenwickTree(ft_size)
        max_sum = -float('inf')
        
        for i in range(n):
            a_i = a_list[i]
            r = bisect.bisect_right(sorted_a, a_i)
            fenwick_rank = r + 1  # Fenwick is 1-based
            
            current_max = fenwick.query(fenwick_rank)
            dp_i = nums[i] + max(current_max, 0)
            
            # Update Fenwick Tree if necessary
            current_val = fenwick.get(fenwick_rank)
            if dp_i > current_val:
                fenwick.update(fenwick_rank, dp_i)
            
            if dp_i > max_sum:
                max_sum = dp_i
        
        return max_sum