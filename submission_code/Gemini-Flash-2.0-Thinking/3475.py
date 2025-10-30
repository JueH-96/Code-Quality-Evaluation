class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        operations = 0
        temp_nums = list(nums)

        for i in range(n - 2):
            if temp_nums[i] == 0:
                operations += 1
                for j in range(i, i + 3):
                    temp_nums[j] = 1 - temp_nums[j]

        for i in range(n):
            if temp_nums[i] == 0:
                return -1

        return operations