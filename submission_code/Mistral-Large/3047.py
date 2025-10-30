from typing import List
from collections import defaultdict
from math import isqrt

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        def prime_factors(n):
            factors = []
            # Check for number of 2s that divide n
            while n % 2 == 0:
                factors.append(2)
                n = n // 2
            # n must be odd at this point so a skip of 2 (i.e., i = i + 2) can be used
            for i in range(3, isqrt(n) + 1, 2):
                # While i divides n, append i and divide n
                while n % i == 0:
                    factors.append(i)
                    n = n // i
            # Condition if n is a prime number greater than 2
            if n > 2:
                factors.append(n)
            return factors

        def factor_signature(n):
            factors = prime_factors(n)
            signature = 0
            for factor in factors:
                signature ^= (1 << factor)
            return signature

        max_sum = 0
        factor_groups = defaultdict(int)

        for num in nums:
            signature = factor_signature(num)
            if signature in factor_groups:
                max_sum = max(max_sum, factor_groups[signature] + num)
            factor_groups[signature] = max(factor_groups[signature], num)

        return max_sum