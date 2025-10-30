class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        operations = 0
        for i in range(n - 1, 0, -1):
            while nums[i] < nums[i - 1]:
                greatest_proper_divisor = 0
                for j in range(1, nums[i - 1]):
                    if nums[i - 1] % j == 0:
                        greatest_proper_divisor = j
                if greatest_proper_divisor == 0:
                    return -1
                nums[i - 1] //= greatest_proper_divisor
                operations += 1
        return operations