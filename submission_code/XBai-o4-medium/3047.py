from typing import List
from collections import defaultdict

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        def square_free(n):
            res = 1
            count = 0
            # Handle factor 2
            while n % 2 == 0:
                count += 1
                n = n // 2
            if count % 2 == 1:
                res *= 2
            # Check odd factors
            i = 3
            while i * i <= n:
                count = 0
                while n % i == 0:
                    count += 1
                    n = n // i
                if count % 2 == 1:
                    res *= i
                i += 2
            # If remaining n is a prime
            if n > 1:
                res *= n
            return res
        
        groups = defaultdict(int)
        max_sum = 0
        for x in nums:
            sf = square_free(x)
            groups[sf] += x
            if groups[sf] > max_sum:
                max_sum = groups[sf]
        return max_sum