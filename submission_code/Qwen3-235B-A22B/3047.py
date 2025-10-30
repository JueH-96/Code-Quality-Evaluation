from typing import List
from collections import defaultdict

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        def square_free_part(n):
            if n == 0:
                return 0
            res = 1
            i = 2
            while i * i <= n:
                count = 0
                while n % i == 0:
                    count += 1
                    n //= i
                if count % 2 == 1:
                    res *= i
                i += 1
            if n > 1:
                res *= n
            return res
        
        groups = defaultdict(int)
        max_sum = 0
        for num in nums:
            sf = square_free_part(num)
            groups[sf] += num
            if groups[sf] > max_sum:
                max_sum = groups[sf]
        return max_sum