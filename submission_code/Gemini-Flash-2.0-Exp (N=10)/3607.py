import math

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        def largest_proper_divisor(n):
            if n == 1:
                return 1
            for i in range(2, int(math.sqrt(n)) + 1):
                if n % i == 0:
                    return n // i
            return 1
        
        operations = 0
        for i in range(len(nums) - 1, 0, -1):
            if nums[i] < nums[i-1]:
                if nums[i] == 1:
                    return -1
                
                while nums[i] < nums[i-1]:
                    nums[i-1] = largest_proper_divisor(nums[i-1])
                    operations += 1
                    if nums[i-1] == 1 and nums[i] < nums[i-1]:
                        return -1
        return operations