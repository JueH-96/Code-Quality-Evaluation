from typing import List

class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        max_len = -1
        n = len(nums)
        for i in range(n - 1):
            # Check if the first two elements meet the condition
            if nums[i + 1] != nums[i] + 1:
                continue
            current_len = 2
            expected_diff = -1
            j = i + 2
            # Check subsequent elements
            while j < n:
                if nums[j] == nums[j - 1] + expected_diff:
                    current_len += 1
                    expected_diff = -expected_diff  # Flip the expected difference
                    j += 1
                else:
                    break
            # Update the maximum length found
            if current_len > max_len:
                max_len = current_len
        return max_len if max_len >= 2 else -1