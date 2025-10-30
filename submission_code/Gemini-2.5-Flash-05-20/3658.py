import math
from typing import List

class Solution:
    def minDifference(self, nums: List[int]) -> int:
        n = len(nums)

        # Step 1: Check if there are any missing elements (-1)
        has_missing = False
        for x in nums:
            if x == -1:
                has_missing = True
                break

        # Step 2: Case 1: No missing elements. Calculate max difference directly.
        if not has_missing:
            max_fixed_diff = 0
            for i in range(n - 1):
                max_fixed_diff = max(max_fixed_diff, abs(nums[i] - nums[i+1]))
            return max_fixed_diff

        # Step 3: Case 2: Missing elements are present.

        # Part A: Calculate max_fixed_diff (differences between adjacent non-missing elements)
        max_fixed_diff = 0
        for i in range(n - 1):
            if nums[i] != -1 and nums[i+1] != -1:
                max_fixed_diff = max(max_fixed_diff, abs(nums[i] - nums[i+1]))
        
        # Part B: Find min_adj_val and max_adj_val among fixed elements adjacent to any -1
        min_adj_val = float('inf')
        max_adj_val = float('-inf')

        for i in range(n):
            if nums[i] == -1:
                # Check left neighbor
                if i > 0 and nums[i-1] != -1:
                    min_adj_val = min(min_adj_val, nums[i-1])
                    max_adj_val = max(max_adj_val, nums[i-1])
                # Check right neighbor
                if i < n - 1 and nums[i+1] != -1:
                    min_adj_val = min(min_adj_val, nums[i+1])
                    max_adj_val = max(max_adj_val, nums[i+1])
        
        # Part C: Handle Subcase 2.1: All elements are -1
        # If min_adj_val is still infinity, it means no fixed elements were adjacent to any -1.
        # Since we already know has_missing is True, this implies all elements must be -1.
        if min_adj_val == float('inf'):
            return 0 
        
        # Part D: Handle Subcase 2.2: There are missing elements and fixed elements adjacent to -1s.
        # Calculate the minimum required difference for filling the gaps (using a single value 'k' as 'x' and 'y')
        # This is equivalent to ceil((max_adj_val - min_adj_val) / 2)
        D_between_fixed_and_missing = (max_adj_val - min_adj_val + 1) // 2

        # The overall minimum difference is the maximum of the two types of differences.
        return max(max_fixed_diff, D_between_fixed_and_missing)