from typing import List

class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        max_len = -1
        n = len(nums)
        for i in range(n - 1):
            if nums[i + 1] - nums[i] != 1:
                continue
            current_len = 2
            next_val = nums[i]  # The next expected value after the initial pair
            j = i + 2
            while j < n:
                if nums[j] == next_val:
                    current_len += 1
                    # Toggle next_val between nums[i] and nums[i]+1
                    next_val = nums[i] if next_val == nums[i] + 1 else nums[i] + 1
                    j += 1
                else:
                    break
            if current_len > max_len:
                max_len = current_len
        return max_len if max_len >= 2 else -1