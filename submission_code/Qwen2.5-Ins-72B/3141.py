from typing import List

class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        total_sum = sum(nums)
        if total_sum == 0:
            return -1
        
        # Calculate the number of full cycles needed
        full_cycles = target // total_sum
        remaining_target = target % total_sum
        
        # Initialize variables for sliding window
        min_length = float('inf')
        current_sum = 0
        left = 0
        
        # Use a sliding window to find the shortest subarray with sum equal to remaining_target
        for right in range(2 * n):
            current_sum += nums[right % n]
            
            while current_sum > remaining_target:
                current_sum -= nums[left % n]
                left += 1
            
            if current_sum == remaining_target:
                min_length = min(min_length, right - left + 1)
        
        # If no valid subarray is found, return -1
        if min_length == float('inf'):
            return -1
        
        # Return the total length including full cycles
        return min_length + full_cycles * n