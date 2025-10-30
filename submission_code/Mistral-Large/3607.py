from typing import List
import math

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        def greatest_proper_divisor(x):
            for i in range(2, int(math.sqrt(x)) + 1):
                if x % i == 0:
                    return x // i
            return 1

        operations = 0
        n = len(nums)

        for i in range(n - 1):
            while nums[i] > nums[i + 1]:
                gpd = greatest_proper_divisor(nums[i])
                if gpd == 1:
                    return -1
                nums[i] //= gpd
                operations += 1

        return operations