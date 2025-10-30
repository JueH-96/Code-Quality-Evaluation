class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        n = len(nums)
        nums.sort()
        left, right = 0, n - 1
        operations = 0
        while left <= right:
            if nums[left] + nums[right] > target:
                nums[right] -= target - nums[left]
                operations += 1
                right -= 1
            else:
                left += 1
                right -= 1
                operations += 1
        return operations