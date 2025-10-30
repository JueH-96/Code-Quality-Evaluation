class Solution:
    def countSubarrays(self, nums: List[int]) -> int:
        count = 0
        for i in range(1, len(nums) - 1):
            if 2 * (nums[i - 1] + nums[i + 1]) == nums[i]:
                count += 1
        return count