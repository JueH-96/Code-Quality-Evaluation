from typing import List

class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0
        count = 0
        drop_index = -1
        for i in range(n):
            next_i = (i + 1) % n
            if nums[i] > nums[next_i]:
                count += 1
                drop_index = i
        if count != 1:
            return -1
        else:
            return n - drop_index - 1