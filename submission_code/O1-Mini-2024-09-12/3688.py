from typing import List
from collections import defaultdict

class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        def kadane(arr: List[int]) -> int:
            max_sum = current = arr[0]
            for num in arr[1:]:
                current = max(num, current + num)
                max_sum = max(max_sum, current)
            return max_sum
        
        # Calculate the original maximum subarray sum
        original_max = kadane(nums)
        result = original_max
        
        # Map each unique number to its indices
        num_indices = defaultdict(list)
        for idx, num in enumerate(nums):
            num_indices[num].append(idx)
        
        # Iterate over each unique number to remove all its occurrences
        for x, indices in num_indices.items():
            max_sum = float('-inf')
            prev = -1
            # Split the array into segments separated by x
            for idx in indices:
                if prev + 1 <= idx - 1:
                    segment = nums[prev + 1:idx]
                    max_sum = max(max_sum, kadane(segment))
                prev = idx
            # Handle the segment after the last occurrence of x
            if prev + 1 < len(nums):
                segment = nums[prev + 1:]
                max_sum = max(max_sum, kadane(segment))
            # Update the result with the maximum subarray sum after removing x
            result = max(result, max_sum)
        
        return result