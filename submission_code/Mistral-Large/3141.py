from typing import List
import collections

class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        # Dictionary to store the cumulative sum and its index
        sum_index_map = {0: -1}
        current_sum = 0
        min_length = float('inf')
        result = -1

        for i, num in enumerate(nums * 2):  # Simulate the infinite array by doubling the nums array
            current_sum += num
            sum_index_map[current_sum] = i

            needed_sum = current_sum - target
            if needed_sum in sum_index_map:
                min_length = min(min_length, i - sum_index_map[needed_sum])

        return min_length if min_length != float('inf') else -1