from typing import List
import math

class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if n < 2:
            return 0  # No subarrays exist
        
        # Compute prefix sums
        prefix = [0] * n
        prefix[0] = nums[0]
        for j in range(1, n):
            prefix[j] = prefix[j-1] + nums[j]
        
        # Initialize hash map with the first element
        hash_map = {nums[0]: 0}
        max_sum = -math.inf
        
        for j in range(1, n):
            # Calculate required nums[i] values
            required_i1 = nums[j] + k
            required_i2 = nums[j] - k
            
            # Check if required nums[i] exist in hash_map
            if required_i1 in hash_map:
                sum1 = prefix[j] - hash_map[required_i1]
                if sum1 > max_sum:
                    max_sum = sum1
            if required_i2 in hash_map:
                sum2 = prefix[j] - hash_map[required_i2]
                if sum2 > max_sum:
                    max_sum = sum2
            
            # Update hash_map with the minimum prefix sum for nums[j]
            if nums[j] in hash_map:
                hash_map[nums[j]] = min(hash_map[nums[j]], prefix[j-1])
            else:
                hash_map[nums[j]] = prefix[j-1]
        
        # If no good subarray is found, return 0
        if max_sum == -math.inf:
            return 0
        else:
            return max_sum