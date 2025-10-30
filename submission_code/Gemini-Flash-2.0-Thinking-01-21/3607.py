import math

class Solution:
    def get_greatest_proper_divisor(self, n):
        if n == 1:
            return 1
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return n // i
        return 1

    def minOperations(self, nums: List[int]) -> int:
        operations = 0
        n = len(nums)
        for i in range(1, n):
            while nums[i] < nums[i-1]:
                if nums[i-1] == 1:
                    return -1
                greatest_proper_divisor = self.get_greatest_proper_divisor(nums[i-1])
                if greatest_proper_divisor == 1:
                    return -1
                nums[i-1] = nums[i-1] // greatest_proper_divisor
                operations += 1
        return operations