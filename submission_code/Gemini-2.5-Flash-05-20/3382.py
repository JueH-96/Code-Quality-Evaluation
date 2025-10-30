import collections
import bisect
from typing import List

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Step 1: Compute L[j] for all j
        # L[j] is the index of the previous element strictly greater than nums[j].
        # If no such element exists, L[j] is -1.
        # This implies that nums[j] is the maximum element in the subarray nums[L[j]+1 : j+1].
        L = [-1] * n
        stack = [] # Stores indices in decreasing order of nums[index] values.
                   # Specifically, stack[k] < stack[k+1] and nums[stack[k]] > nums[stack[k+1]]
        
        for j in range(n):
            # Pop elements from the stack whose values are less than or equal to nums[j].
            # These elements cannot be the Previous Greater Element (PGE) for nums[j].
            while stack and nums[stack[-1]] <= nums[j]:
                stack.pop()
            
            # If the stack is not empty, the top element is the index of the PGE.
            if stack:
                L[j] = stack[-1]
            
            # Push the current index onto the stack.
            stack.append(j)
            
        # Step 2: Iterate through nums and count valid subarrays
        # A subarray nums[i...j] is valid if nums[i] == nums[j] == max(nums[i...j]).
        # The condition max(nums[i...j]) == nums[j] means that for any k in [i, j-1], nums[k] <= nums[j].
        # From Step 1, this condition is satisfied if i > L[j].
        # So, we need to count pairs (i, j) such that L[j] < i <= j and nums[i] == nums[j].
        
        ans = 0
        # val_indices[value] stores a sorted list of indices where 'value' has appeared previously.
        val_indices = collections.defaultdict(list)
        
        for j in range(n):
            current_val = nums[j]
            
            # Every single-element subarray [nums[j]] (where i=j) is always valid.
            # nums[j] == nums[j] == max([nums[j]])
            ans += 1
            
            # Count 'i' such that L[j] < i < j and nums[i] == current_val.
            # The indices for current_val are in val_indices[current_val].
            # We need to find how many of these indices are greater than L[j].
            
            # bisect_left finds the insertion point for (L[j] + 1) in val_indices[current_val].
            # This index (start_idx_in_list) points to the first element in the list
            # that is greater than L[j].
            # All elements from start_idx_in_list to the end of the list satisfy i > L[j].
            start_idx_in_list = bisect.bisect_left(val_indices[current_val], L[j] + 1)
            
            # The number of such valid 'i's (where i < j) is the count of elements
            # from start_idx_in_list to the end of the list.
            ans += len(val_indices[current_val]) - start_idx_in_list
            
            # Add the current index 'j' to the list of indices for current_val.
            val_indices[current_val].append(j)
            
        return ans