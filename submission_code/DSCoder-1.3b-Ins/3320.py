class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        nums.sort()
        left, right = 0, len(nums) - 1
        operations = 0
        while left < right:
            if nums[left] + nums[right] > 5:
                operations += right - left
                right -= 1
            else:
                left += 1
        return operations