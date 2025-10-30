class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        if len(nums) < 2:
            return 0
        target_sum = nums[0] + nums[1]
        count = 0
        for i in range(len(nums) // 2):
            current_sum = nums[2 * i] + nums[2 * i + 1]
            if current_sum == target_sum:
                count += 1
            else:
                break
        return count