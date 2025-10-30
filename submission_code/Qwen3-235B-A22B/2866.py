from typing import List

class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        max_len = 0
        n = len(nums)
        for i in range(n):
            # Check if the starting element meets the criteria
            if nums[i] % 2 != 0 or nums[i] > threshold:
                continue
            # Initialize current length and previous parity
            current_length = 1
            prev_parity = nums[i] % 2
            max_len = max(max_len, current_length)
            # Extend the subarray as far right as possible
            for j in range(i + 1, n):
                if nums[j] > threshold:
                    break  # Element exceeds threshold, cannot extend further
                current_parity = nums[j] % 2
                if current_parity == prev_parity:
                    break  # Parity not alternating, stop here
                prev_parity = current_parity
                current_length += 1
                max_len = max(max_len, current_length)
        return max_len