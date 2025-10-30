from typing import List

class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        min_prefix = {}
        previous_prefix = 0
        max_sum = float('-inf')
        
        for num in nums:
            current_prefix = previous_prefix + num
            target1 = num + k
            target2 = num - k
            current_max_candidate = float('-inf')
            
            # Check for target1 in min_prefix
            if target1 in min_prefix:
                candidate = current_prefix - min_prefix[target1]
                if candidate > current_max_candidate:
                    current_max_candidate = candidate
            # Check for target2 in min_prefix
            if target2 in min_prefix:
                candidate = current_prefix - min_prefix[target2]
                if candidate > current_max_candidate:
                    current_max_candidate = candidate
            
            # Update the maximum sum if a valid candidate is found
            if current_max_candidate != float('-inf'):
                if current_max_candidate > max_sum:
                    max_sum = current_max_candidate
            
            # Update the min_prefix for the current number
            current_key = num
            if current_key in min_prefix:
                if previous_prefix < min_prefix[current_key]:
                    min_prefix[current_key] = previous_prefix
            else:
                min_prefix[current_key] = previous_prefix
            
            # Move to the next element
            previous_prefix = current_prefix
        
        return max_sum if max_sum != float('-inf') else 0