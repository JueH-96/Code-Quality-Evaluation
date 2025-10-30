import bisect
from collections import defaultdict
from typing import List

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        # Compute next greater index to the right
        next_greater = [n] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] < nums[i]:
                idx = stack.pop()
                next_greater[idx] = i
            stack.append(i)
        
        # Build a dictionary mapping each value to its sorted list of indices
        val_to_indices = defaultdict(list)
        for i in range(n):
            val_to_indices[nums[i]].append(i)  # Indices are added in order, so the list is sorted
        
        # Initialize answer
        ans = 0
        
        # For each starting index i, compute the number of valid ending indices j
        for i in range(n):
            val = nums[i]
            idx_list = val_to_indices[val]
            R_i = next_greater[i] - 1  # The rightmost index where max is nums[i]
            # Count the number of indices j in idx_list with i <= j <= R_i
            count = bisect.bisect_right(idx_list, R_i) - bisect.bisect_left(idx_list, i)
            ans += count
        
        return ans