from typing import List
from collections import defaultdict

class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        # Compute the original maximum subarray sum using Kadane's algorithm
        if not nums:
            return 0
        max_original = current = nums[0]
        for num in nums[1:]:
            current = max(num, current + num)
            max_original = max(max_original, current)
        
        # Create a frequency map to track occurrences of each element
        freq = defaultdict(int)
        for num in nums:
            freq[num] += 1
        
        max_deleted = -float('inf')
        # Iterate through each unique element in the frequency map
        for x in freq:
            # Skip if removing x would result in an empty array
            if freq[x] == len(nums):
                continue
            # Create the new array with all occurrences of x removed
            new_nums = [num for num in nums if num != x]
            # Compute the maximum subarray sum for the new array using Kadane's algorithm
            if not new_nums:
                continue
            current_sum = max_sum = new_nums[0]
            for num in new_nums[1:]:
                current_sum = max(num, current_sum + num)
                max_sum = max(max_sum, current_sum)
            # Update the maximum deleted sum
            max_deleted = max(max_deleted, max_sum)
        
        # Return the maximum of the original and the best deleted sum
        return max(max_original, max_deleted)