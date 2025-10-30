from typing import List
import math

class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        sum_so_far = 0
        prefix_map = {}
        max_sum = -math.inf
        
        for num in nums:
            sum_so_far += num
            
            target1 = num + k
            target2 = num - k
            
            if target1 in prefix_map:
                current_sum = sum_so_far - prefix_map[target1]
                max_sum = max(max_sum, current_sum)
            
            if target2 in prefix_map:
                current_sum = sum_so_far - prefix_map[target2]
                max_sum = max(max_sum, current_sum)
            
            current_prefix = sum_so_far - num
            if num in prefix_map:
                if current_prefix < prefix_map[num]:
                    prefix_map[num] = current_prefix
            else:
                prefix_map[num] = current_prefix
        
        return max_sum if max_sum != -math.inf else 0