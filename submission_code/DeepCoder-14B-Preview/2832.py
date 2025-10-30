from typing import List

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        max_len = 0
        unique = set(nums)
        for v in unique:
            count_non_v = 0
            left = 0
            for right in range(len(nums)):
                if nums[right] != v:
                    count_non_v += 1
                while count_non_v > k:
                    if nums[left] != v:
                        count_non_v -= 1
                    left += 1
                current_v_count = (right - left + 1) - count_non_v
                if current_v_count > max_len:
                    max_len = current_v_count
        return max_len