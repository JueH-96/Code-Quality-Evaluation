from typing import List

class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        max_len = 0
        n = len(nums)
        for l in range(n):
            # Check if the starting element is valid (even and <= threshold)
            if nums[l] % 2 != 0 or nums[l] > threshold:
                continue
            current_length = 1
            prev_parity = 0  # since nums[l] is even
            # Extend the subarray as far right as possible
            for i in range(l + 1, n):
                current_num = nums[i]
                if current_num > threshold:
                    break
                current_parity = current_num % 2
                if current_parity != prev_parity:
                    current_length += 1
                    prev_parity = current_parity
                else:
                    break  # parity alternation failed
            # Update the maximum length found
            if current_length > max_len:
                max_len = current_length
        return max_len