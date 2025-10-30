from bisect import bisect_left
from typing import List

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        # Step 1: Compute next greater element for each index.
        # For every index i, next_greater[i] is the first index j > i such that nums[j] > nums[i]
        # If no such index exists, set next_greater[i] = n.
        next_greater = [n] * n
        stack = []  # will store indices in a decreasing order of nums[.]
        for i in range(n):
            while stack and nums[i] > nums[stack[-1]]:
                idx = stack.pop()
                next_greater[idx] = i
            stack.append(i)
        # No need to process the remaining indices in stack as their next_greater is already n.
        
        # Step 2: Build a mapping from a number to the list of indices where it appears
        occurrences = {}
        for i, num in enumerate(nums):
            if num not in occurrences:
                occurrences[num] = []
            occurrences[num].append(i)
            
        # Step 3: For each starting index i, count the number of valid subarrays starting at i.
        # A subarray starting at index i and ending at some j (i <= j < next_greater[i]) is valid if nums[j] == nums[i].
        # This is because when all elements in the subarray are <= nums[i] and the endpoints are nums[i],
        # the maximum of the subarray is nums[i].
        total = 0
        for i in range(n):
            v = nums[i]
            # valid j's are in the range [i, next_greater[i]) and must satisfy nums[j] == v.
            # Since occurrences[v] is sorted, we can use binary search.
            pos_list = occurrences[v]
            hi = bisect_left(pos_list, next_greater[i])
            lo = bisect_left(pos_list, i)
            count = hi - lo
            total += count
        
        return total