from typing import List

class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        current_sum = nums[0]
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1] + 1:
                current_sum += nums[i]
            else:
                break
        sum_sp = current_sum
        num_set = set(nums)
        x = sum_sp
        while True:
            if x not in num_set:
                return x
            x += 1