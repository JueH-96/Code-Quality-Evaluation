from bisect import bisect_right
import sys
from typing import List

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Compute a_i = nums[i] - i for each i
        a = [nums[i] - i for i in range(n)]
        dummy = -sys.float_info.max  # Represents the empty subsequence's a_j
        
        # Collect all a_i and add the dummy to prepare for coordinate compression
        all_a = a + [dummy]
        unique_a = sorted(list(set(all_a)))
        
        # Create a rank mapping for each unique a value
        rank = {v: i for i, v in enumerate(unique_a)}
        size = len(unique_a)
        
        # Fenwick Tree implementation for maximum queries and updates
        class FenwickTree:
            def __init__(self, size):
                self.n = size
                self.tree = [-sys.float_info.max for _ in range(self.n + 1)]  # 1-based indexing
            
            def update(self, idx, value):
                # Update the Fenwick Tree to have the maximum value at idx (1-based)
                while idx <= self.n:
                    if value > self.tree[idx]:
                        self.tree[idx] = value
                    else:
                        break  # No need to proceed if no update is needed
                    idx += idx & -idx
            
            def query(self, idx):
                # Query the maximum value from 1 to idx (1-based)
                res = -sys.float_info.max
                while idx > 0:
                    if self.tree[idx] > res:
                        res = self.tree[idx]
                    idx -= idx & -idx
                return res
        
        ft = FenwickTree(size)
        dummy_rank = rank[dummy]
        ft.update(dummy_rank + 1, 0)  # Initialize the dummy's position with sum 0
        
        global_max = -sys.float_info.max
        
        for i in range(n):
            current_a = a[i]
            # Find the rank of current_a in the sorted unique list
            r = bisect_right(unique_a, current_a) - 1
            # Query the Fenwick Tree up to rank r (converted to 1-based index)
            max_prev = ft.query(r + 1)
            current_sum = max_prev + nums[i]
            
            # Update the global maximum
            if current_sum > global_max:
                global_max = current_sum
            
            # Update the Fenwick Tree at the current rank if current_sum is larger
            existing_value = ft.query(r + 1)
            if current_sum > existing_value:
                ft.update(r + 1, current_sum)
        
        return global_max