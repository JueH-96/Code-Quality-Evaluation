class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        count = 0
        for i in range(len(nums) - 1):
            if nums[i] & nums[i + 1] < nums[i + 1]:
                count += 1
        return count + 1