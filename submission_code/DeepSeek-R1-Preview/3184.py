from bisect import bisect_left
from typing import List

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [-float('inf')] * (self.n + 1)  # 1-based indexing

    def update(self, index, value):
        while index <= self.n:
            if self.tree[index] < value:
                self.tree[index] = value
            else:
                break
            index += index & -index

    def query(self, index):
        res = -float('inf')
        while index > 0:
            if self.tree[index] > res:
                res = self.tree[index]
            index -= index & -index
        return res

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        if not nums:
            return 0
        n = len(nums)
        # Compute the transformed keys
        keys = [nums[i] - i for i in range(n)]
        # Extract unique keys and sort them to assign ranks
        unique_keys = sorted(list(set(keys)))
        ft_size = len(unique_keys)
        ft = FenwickTree(ft_size)
        max_sum = -float('inf')
        for i in range(n):
            key_i = keys[i]
            # Find the rank using bisect_left
            rank = bisect_left(unique_keys, key_i) + 1  # 1-based rank
            # Query the Fenwick Tree for maximum sum up to current rank
            q = ft.query(rank)
            current_sum = max(q + nums[i], nums[i])
            # Update the Fenwick Tree with the current_sum at this rank
            ft.update(rank, current_sum)
            # Update the maximum sum found so far
            if current_sum > max_sum:
                max_sum = current_sum
        return max_sum