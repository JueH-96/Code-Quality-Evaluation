from typing import List

class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        prefix_sum = 0
        prefix_map = {}
        max_sum = -float('inf')
        
        for current_num in nums:
            # Check for current_num - k and current_num + k in the map
            target1 = current_num - k
            target2 = current_num + k
            
            current_prefix = prefix_sum + current_num  # This is prefix[j+1]
            
            # Check if target1 exists in the map
            if target1 in prefix_map:
                current_sum = current_prefix - prefix_map[target1]
                if current_sum > max_sum:
                    max_sum = current_sum
            # Check if target2 exists in the map
            if target2 in prefix_map:
                current_sum = current_prefix - prefix_map[target2]
                if current_sum > max_sum:
                    max_sum = current_sum
            
            # Update the prefix_map with the current_num and the minimum prefix_sum
            if current_num in prefix_map:
                if prefix_sum < prefix_map[current_num]:
                    prefix_map[current_num] = prefix_sum
            else:
                prefix_map[current_num] = prefix_sum
            
            # Update prefix_sum for the next iteration
            prefix_sum += current_num
        
        return max_sum if max_sum != -float('inf') else 0