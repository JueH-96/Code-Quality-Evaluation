import sys

class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        # Compute the original maximum subarray sum using Kadane's algorithm
        max_current = max_global = nums[0]
        for num in nums[1:]:
            max_current = max(num, max_current + num)
            max_global = max(max_global, max_current)
        
        # Compute the maximum subarray sum for each possible x
        max_after_remove = max_global
        
        # Create a dictionary to map each x to its indices
        x_to_indices = {}
        for idx, num in enumerate(nums):
            if num not in x_to_indices:
                x_to_indices[num] = []
            x_to_indices[num].append(idx)
        
        # For each unique x, compute the maximum subarray sum after removing x
        for x in x_to_indices:
            indices = x_to_indices[x]
            # Create a new array without x
            new_nums = [num for idx, num in enumerate(nums) if idx not in indices]
            # Compute max subarray sum for new_nums using Kadane's algorithm
            if not new_nums:
                current_max = 0
            else:
                current_max = new_nums[0]
                for num in new_nums[1:]:
                    current_max = max(num, current_max + num)
                max_subarray = max(new_nums) if not new_nums else max_subarray
                # Correctly compute max_subarray
                current_max = new_nums[0]
                for num in new_nums[1:]:
                    current_max = max(num, current_max + num)
                max_subarray = current_max
            
            if max_subarray > max_after_remove:
                max_after_remove = max_subarray
        
        return max_after_remove