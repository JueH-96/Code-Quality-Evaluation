from typing import List

class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        # Dictionary to store the first occurrence of each cumulative sum
        sum_index = {}
        current_sum = 0
        max_sum = 0

        for i, num in enumerate(nums):
            current_sum += num

            # Check if there is a previous sum that forms a good subarray
            if current_sum - k in sum_index:
                max_sum = max(max_sum, current_sum - sum_index[current_sum - k])

            # Check if there is a previous sum that forms a good subarray in the reverse direction
            if current_sum + k in sum_index:
                max_sum = max(max_sum, current_sum - sum_index[current_sum + k])

            # Store the first occurrence of the current sum
            if current_sum not in sum_index:
                sum_index[current_sum] = current_sum

        return max_sum if max_sum != 0 else 0