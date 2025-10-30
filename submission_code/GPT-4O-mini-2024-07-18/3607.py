from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        def greatest_proper_divisor(x: int) -> int:
            if x <= 1:
                return 0
            for i in range(int(x**0.5), 0, -1):
                if x % i == 0:
                    if i < x:
                        return i
                    if (x // i) < x:
                        return x // i
            return 0
        
        operations = 0
        n = len(nums)
        
        for i in range(n - 1, 0, -1):
            if nums[i] < nums[i - 1]:
                while nums[i] < nums[i - 1]:
                    gpd = greatest_proper_divisor(nums[i - 1])
                    if gpd == 0:
                        return -1
                    nums[i - 1] //= gpd
                    operations += 1
        
        return operations