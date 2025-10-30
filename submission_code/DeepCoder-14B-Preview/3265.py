from typing import List

class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if n < 2:
            return 0
        
        # Compute the prefix sum array
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        value_to_min_prefix = {}
        max_sum = float('-inf')
        
        for j in range(n):
            current_sum = prefix[j + 1]
            
            # Calculate the two possible target values
            v1 = nums[j] + k
            v2 = nums[j] - k
            
            # Check for v1
            if v1 in value_to_min_prefix:
                sum_candidate = current_sum - value_to_min_prefix[v1]
                if sum_candidate > max_sum:
                    max_sum = sum_candidate
            
            # Check for v2
            if v2 in value_to_min_prefix:
                sum_candidate = current_sum - value_to_min_prefix[v2]
                if sum_candidate > max_sum:
                    max_sum = sum_candidate
            
            # Add the current value and its prefix to the dictionary
            current_value = nums[j]
            current_prefix = prefix[j]  # sum up to j (exclusive)
            
            if current_value in value_to_min_prefix:
                if current_prefix < value_to_min_prefix[current_value]:
                    value_to_min_prefix[current_value] = current_prefix
            else:
                value_to_min_prefix[current_value] = current_prefix
        
        # Return the result based on whether any good subarray was found
        return max_sum if max_sum != float('-inf') else 0