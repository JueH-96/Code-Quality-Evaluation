import math
from typing import List

class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        current_prefix = 0
        value_map = dict()
        max_sum = -math.inf
        for num in nums:
            prefix_j_plus_1 = current_prefix + num
            # Check for target1: num - k
            target1 = num - k
            if target1 in value_map:
                candidate = prefix_j_plus_1 - value_map[target1]
                if candidate > max_sum:
                    max_sum = candidate
            # Check for target2: num + k
            target2 = num + k
            if target2 in value_map:
                candidate = prefix_j_plus_1 - value_map[target2]
                if candidate > max_sum:
                    max_sum = candidate
            # Update the value_map with current num's prefix
            if num in value_map:
                if current_prefix < value_map[num]:
                    value_map[num] = current_prefix
            else:
                value_map[num] = current_prefix
            # Move to next prefix
            current_prefix = prefix_j_plus_1
        return max_sum if max_sum != -math.inf else 0