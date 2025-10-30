import math
from typing import List

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        # Compute val = nums[i] - i for all i
        vals = [num - i for i, num in enumerate(nums)]
        # Get sorted unique values for ranking
        unique_vals = sorted(set(vals))
        len_unique = len(unique_vals)
        # Create a rank map from value to rank (1-based for Fenwick tree)
        rank_map = {val: idx + 1 for idx, val in enumerate(unique_vals)}
        
        # Initialize Fenwick tree with negative infinity
        ft = [float('-inf')] * (len_unique + 1)
        
        # Initialize answer to negative infinity
        ans = float('-inf')
        
        # Define update function for Fenwick tree
        def update(idx, val):
            while idx <= len_unique:
                ft[idx] = max(ft[idx], val)
                idx += idx & -idx  # Update index using low-bit manipulation
        
        # Define query function for prefix maximum up to idx
        def query(idx):
            res = float('-inf')
            idx_val = idx
            while idx_val > 0:
                res = max(res, ft[idx_val])
                idx_val -= idx_val & -idx_val  # Move to parent using low-bit manipulation
            return res
        
        # Iterate through each element in nums
        for i in range(n):
            val_i = vals[i]
            rank_i = rank_map[val_i]
            # Query the maximum dp value for previous indices with val <= val_i
            max_prev = query(rank_i)
            # Compute dp_i with the option to start a new subsequence
            dp_i = nums[i] + max(0, max_prev)
            # Update the answer with the current dp_i
            ans = max(ans, dp_i)
            # Update the Fenwick tree with the current dp_i at the appropriate rank
            update(rank_i, dp_i)
        
        # The answer should be an integer, cast and return
        return int(ans)