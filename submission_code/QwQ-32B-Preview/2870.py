from typing import List

class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        max_length = -1
        n = len(nums)
        for i in range(n - 1):
            if nums[i + 1] - nums[i] == 1:
                current_length = 2
                expected_diff = -1
                k = i + 2
                while k < n:
                    if nums[k] - nums[k - 1] == expected_diff:
                        current_length += 1
                        expected_diff *= -1
                        k += 1
                    else:
                        break
                if current_length >= 2:
                    if current_length > max_length:
                        max_length = current_length
        return max_length