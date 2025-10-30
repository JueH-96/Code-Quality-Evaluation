from typing import List
from collections import defaultdict
from bisect import bisect_left, bisect_right

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        
        # 1. Calculate next_greater_equal array
        # next_greater_equal[i] = smallest index k > i with nums[k] > nums[i]
        # This is calculated using a monotonic decreasing stack.
        # Iterate from right to left.
        next_greater_equal = [n] * n
        stack = [] # Stores indices j such that nums[j] is decreasing (from right to left)
        
        for i in range(n - 1, -1, -1):
            # Pop indices from stack whose corresponding values are less than or equal to nums[i]
            # These indices cannot be the next greater element to the right of i
            # because nums[i] is greater than or equal to them and is to their left.
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
                
            # If the stack is not empty, the top element is the index of the first element
            # to the right of i that is strictly greater than nums[i].
            if stack:
                next_greater_equal[i] = stack[-1]
            
            # Push the current index onto the stack.
            stack.append(i)

        # 2. Create a dictionary mapping value to a sorted list of indices where it appears.
        # This allows quick lookup of indices for a specific value.
        indices = defaultdict(list)
        for i, num in enumerate(nums):
            indices[num].append(i)
        
        # 3. Count valid subarrays.
        total_count = 0
        
        # Iterate through each index i. Consider it as the potential *left* boundary
        # of a valid subarray nums[i:j+1].
        for i in range(n):
            v = nums[i]
            
            # A subarray nums[i:j+1] is valid if nums[i] == nums[j] == max(nums[i:j+1]).
            # This implies that all elements nums[k] for i <= k <= j must be <= v.
            # The condition max(nums[i:j+1]) == v means there is no element k in [i, j]
            # such that nums[k] > v.
            # The first index k > i where nums[k] > v is next_greater_equal[i].
            # Therefore, any valid right boundary j for a subarray starting at i
            # must be in the range [i, next_greater_equal[i] - 1].
            R = next_greater_equal[i] - 1
            
            # We need to count indices j such that i <= j <= R and nums[j] == v.
            # These indices j must belong to the precomputed list of indices for value v,
            # which is indices[v]. This list is sorted.
            idx_list = indices[v]
            
            # Find the index k of i in the sorted list idx_list.
            # bisect_left finds the insertion point for i, ensuring we find the correct index
            # even if the value i appears multiple times (it finds the index of the first element >= i,
            # which is the occurrence of i we are currently processing).
            k = bisect_left(idx_list, i)

            # We are looking for indices j in idx_list such that j is at index k or later (i.e., j >= i)
            # AND j <= R.
            # In the sorted list idx_list, indices >= i are at positions k, k+1, ...
            # We need to find how many of these indices are <= R.
            # bisect_right finds the insertion point for R, which is the index of the first element > R.
            # All elements in idx_list at indices 0, ..., q-1 are <= R.
            # We want elements from index k up to q-1.
            q = bisect_right(idx_list, R)
            
            # The number of such indices j is q - k.
            # Each such index j forms a valid subarray nums[i:j+1] starting at i.
            total_count += (q - k)

        return total_count