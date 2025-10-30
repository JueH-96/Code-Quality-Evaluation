from typing import List
import math, collections

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        def sieve(n):
            is_prime = [True] * (n+1)
            is_prime[0] = is_prime[1] = False
            for i in range(2, int(math.sqrt(n)) + 1):
                if is_prime[i]:
                    for j in range(i*i, n+1, i):
                        is_prime[j] = False
            primes = [i for i, val in enumerate(is_prime) if val]
            return primes

        primes = sieve(32000)

        def get_square_free(x):
            res = 1
            for p in primes:
                if p*p > x:
                    break
                count = 0
                while x % p == 0:
                    count += 1
                    x = x // p
                if count % 2 == 1:
                    res *= p
            if x > 1:
                res *= x
            return res

        cache = {}
        group = collections.defaultdict(int)
        for num in nums:
            if num in cache:
                key = cache[num]
            else:
                key = get_square_free(num)
                cache[num] = key
            group[key] += num

        return max(group.values())