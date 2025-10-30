from typing import List

class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        max_len = 0
        n = len(nums)
        for l in range(n):
            if nums[l] % 2 == 0 and nums[l] <= threshold:
                current_len = 1
                if current_len > max_len:
                    max_len = current_len
                prev_parity = nums[l] % 2
                for r in range(l + 1, n):
                    if nums[r] > threshold:
                        break
                    current_parity = nums[r] % 2
                    if current_parity == prev_parity:
                        break
                    current_len += 1
                    prev_parity = current_parity
                    if current_len > max_len:
                        max_len = current_len
        return max_len