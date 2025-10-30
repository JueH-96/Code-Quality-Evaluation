from typing import List
from collections import defaultdict
import math

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        def prime_factors(n):
            factors = defaultdict(int)
            d = 2
            while d * d <= n:
                while (n % d) == 0:
                    factors[d] += 1
                    n //= d
                d += 1
            if n > 1:
                factors[n] += 1
            return factors
        
        def get_key(num):
            factors = prime_factors(num)
            key = []
            for p, exp in factors.items():
                if exp % 2 != 0:
                    key.append(p)
            return tuple(sorted(key))
        
        groups = defaultdict(int)
        
        for num in nums:
            key = get_key(num)
            groups[key] += num
        
        return max(groups.values(), default=0)