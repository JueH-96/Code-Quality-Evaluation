from typing import List

class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        max_len = -1
        n = len(nums)
        for i in range(n - 1):
            if nums[i+1] - nums[i] != 1:
                continue
            current_len = 2
            max_len = max(max_len, current_len)
            expected_diff = -1
            for j in range(i + 2, n):
                if nums[j] - nums[j-1] == expected_diff:
                    current_len += 1
                    max_len = max(max_len, current_len)
                    expected_diff *= -1
                else:
                    break
        return max_len