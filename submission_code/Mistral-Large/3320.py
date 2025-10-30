from typing import List

class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        count = 0
        while len(nums) >= 2:
            score = nums[0] + nums[1]
            nums = nums[2:]
            current_count = 1

            while len(nums) >= 2 and nums[0] + nums[1] == score:
                nums = nums[2:]
                current_count += 1

            count = max(count, current_count)

        return count