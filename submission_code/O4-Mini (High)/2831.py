from typing import List
import math

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        def first_digit(x: int) -> int:
            # Extract the first digit of x
            while x >= 10:
                x //= 10
            return x
        
        count = 0
        n = len(nums)
        
        for i in range(n):
            fd = first_digit(nums[i])
            for j in range(i + 1, n):
                ld = nums[j] % 10
                if math.gcd(fd, ld) == 1:
                    count += 1
        
        return count