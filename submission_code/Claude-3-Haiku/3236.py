class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        nums.sort()
        max_prefix_sum = 0
        for i in range(len(nums)):
            if i == 0 or nums[i] == nums[i-1] + 1:
                max_prefix_sum += nums[i]
            else:
                return max_prefix_sum + 1
        return max_prefix_sum + 1