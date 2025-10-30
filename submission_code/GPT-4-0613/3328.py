class Solution:
    def minOperations(self, k: int) -> int:
        if k == 1:
            return 0
        nums = [1]
        operations = 0
        while sum(nums) < k:
            nums.sort()
            if nums[-1] * 2 > sum(nums):
                nums.append(nums[-1])
                operations += 1
            else:
                nums[-1] += 1
                operations += 1
        return operations