class Solution:
    def minOperations(self, nums: List[int]) -> int:
        def greatest_proper_divisor(n):
            for i in range(n // 2, 0, -1):
                if n % i == 0:
                    return i
            return 1

        n = len(nums)
        operations = 0

        for i in range(1, n):
            while nums[i] < nums[i - 1]:
                nums[i] = greatest_proper_divisor(nums[i])
                operations += 1
                if nums[i] == 1:
                    return -1

        return operations