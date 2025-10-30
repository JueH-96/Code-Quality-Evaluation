class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        n = len(nums)
        nums.sort()
        target.sort()
        operations = 0
        for i in range(n):
            if nums[i] < target[i]:
                operations += target[i] - nums[i]
            elif nums[i] > target[i]:
                operations += nums[i] - target[i]
        return operations