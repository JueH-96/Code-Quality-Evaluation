from typing import List

class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        max_length = 0
        n = len(nums)
        for l in range(n):
            # Check if the starting element is even and within threshold
            if nums[l] % 2 != 0 or nums[l] > threshold:
                continue
            current_length = 1
            prev_parity = 0  # 0 for even, 1 for odd
            for r in range(l + 1, n):
                if nums[r] > threshold:
                    break  # Exceeds threshold, stop here
                current_parity = nums[r] % 2
                if current_parity == prev_parity:
                    break  # Parity doesn't alternate, stop here
                prev_parity = current_parity
                current_length += 1
            if current_length > max_length:
                max_length = current_length
        return max_length