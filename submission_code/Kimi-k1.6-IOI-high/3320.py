class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        if len(nums) < 2:
            return 0
        target_sum = nums[0] + nums[1]
        count = 1
        i = 2
        while i + 1 < len(nums):
            if nums[i] + nums[i + 1] == target_sum:
                count += 1
                i += 2
            else:
                break
        return count