from typing import List
import bisect

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [float('-inf')] * (self.n + 2)  # 1-based indexing

    def update(self, idx, value):
        # idx is 1-based
        while idx <= self.n:
            if value > self.tree[idx]:
                self.tree[idx] = value
            else:
                break  # No need to proceed further if no update is needed
            idx += idx & -idx

    def query(self, idx):
        # Returns maximum value in [1, idx]
        res = float('-inf')
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
        
        # Compute a[i] = nums[i] - i for each element
        a = [nums[i] - i for i in range(n)]
        
        # Coordinate compression for a values
        unique_a = sorted(set(a))
        rank_map = {v: i+1 for i, v in enumerate(unique_a)}  # Using 1-based ranking
        max_rank = len(unique_a)
        
        # Initialize Fenwick Tree
        ft = FenwickTree(max_rank)
        max_sum = float('-inf')
        
        for i in range(n):
            ai = a[i]
            r = rank_map[ai]
            prev_max = ft.query(r)
            current_sum = nums[i] + max(prev_max, 0)
            
            # Update Fenwick Tree
            ft.update(r, current_sum)
            
            # Update the global maximum sum
            if current_sum > max_sum:
                max_sum = current_sum
        
        return max_sum