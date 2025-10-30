from typing import List
import math
from collections import defaultdict

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        # First, build a list of primes up to sqrt(max(nums))
        # so we can factor each number quickly.
        max_num = max(nums)
        limit = int(math.isqrt(max_num)) + 1
        
        # Simple sieve to generate primes up to 'limit'
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        primes = []
        for i in range(2, limit + 1):
            if sieve[i]:
                primes.append(i)
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        
        def square_free_part(x: int) -> int:
            """
            Return the square-free part of x: the product of
            primes dividing x to an odd exponent.
            """
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
                    # If exponent is odd, include p once
                    if cnt % 2 == 1:
                        res *= p
                # continue to next prime
            # if there's a remaining prime factor > sqrt(x), it appears once
            if tmp > 1:
                # that leftover prime factor has exponent 1
                res *= tmp
            return res
        
        # Group numbers by their square-free part
        group_sum = defaultdict(int)
        ans = 0
        
        for v in nums:
            sf = square_free_part(v)
            group_sum[sf] += v
            # track the best seen so far
            ans = max(ans, group_sum[sf])
            # also a single element might be better (though with positives it won't)
            ans = max(ans, v)
        
        return ans