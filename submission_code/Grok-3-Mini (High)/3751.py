from typing import List
import math

class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Compute original count of k
        orig_k = sum(1 for num in nums if num == k)
        # Max gain over all v
        max_gain = 0
        for v in range(1, 51):  # v from 1 to 50
            # Kadane for diff array: [num==v] - [num==k]
            max_sum_v = float('-inf')
            current_sum = 0
            for num in nums:
                diff_val = (1 if num == v else 0) - (1 if num == k else 0)
                current_sum = max(diff_val, current_sum + diff_val)
                if current_sum > max_sum_v:
                    max_sum_v = current_sum
            # Update max_gain with the max subarray sum for this v
            if max_sum_v > max_gain:
                max_gain = max_sum_v
        # The maximum frequency is original count plus the maximum gain
        return orig_k + max_gain