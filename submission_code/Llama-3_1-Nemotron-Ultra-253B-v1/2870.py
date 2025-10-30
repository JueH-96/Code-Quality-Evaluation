from typing import List

class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        max_len = -1
        n = len(nums)
        for i in range(n - 1):
            if nums[i + 1] != nums[i] + 1:
                continue
            current_len = 2
            expected_diff = -1
            j = i + 2
            while j < n:
                if nums[j] != nums[j - 1] + expected_diff:
                    break
                current_len += 1
                expected_diff *= -1
                j += 1
            if current_len > max_len:
                max_len = current_len
        return max_len if max_len >= 2 else -1