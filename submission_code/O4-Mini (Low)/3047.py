from typing import List
import math

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        # first, build a list of primes up to sqrt(1e9) ~ 31623
        # we'll sieve up to 32000
        MAXP = 32000
        is_prime = [True] * (MAXP + 1)
        is_prime[0] = is_prime[1] = False
        primes = []
        for i in range(2, MAXP + 1):
            if is_prime[i]:
                primes.append(i)
                for j in range(i*i, MAXP + 1, i):
                    is_prime[j] = False
        
        def squarefree(x: int) -> int:
            # reduce x to its squarefree kernel: product of primes
            # that appear with odd exponent
            res = 1
            tmp = x
            for p in primes:
                if p * p > tmp:
                    break
                if tmp % p == 0:
                    cnt = 0
                    while tmp % p == 0:
                        tmp //= p
                        cnt += 1
                    if cnt % 2 == 1:
                        res *= p
                # if tmp becomes 1, we can stop
            if tmp > 1:
                # the remainder is prime, exponent exactly 1
                res *= tmp
            return res
        
        best = 0
        group_sum = {}
        for v in nums:
            key = squarefree(v)
            group_sum[key] = group_sum.get(key, 0) + v
            if group_sum[key] > best:
                best = group_sum[key]
            # also track single element in case that's larger
            if v > best:
                best = v
        
        return best