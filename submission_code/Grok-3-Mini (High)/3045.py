from typing import List

class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        drop_count = 0
        drop_idx = -1
        for i in range(n - 1):
            if nums[i] > nums[i + 1]:
                drop_count += 1
                drop_idx = i
        if drop_count > 1:
            return -1
        elif drop_count == 0:
            return 0
        else:  # drop_count == 1
            max_val = max(nums)
            min_val = min(nums)
            if nums[drop_idx] == max_val and nums[drop_idx + 1] == min_val:
                min_idx = drop_idx + 1
                k = (n - min_idx) % n
                return k
            else:
                return -1