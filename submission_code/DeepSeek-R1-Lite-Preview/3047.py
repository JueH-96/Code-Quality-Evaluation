import math
from typing import List

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        # Step 1: Generate all primes up to sqrt(10^9) using Sieve of Eratosthenes
        def sieve(limit):
            is_prime = [True] * (limit + 1)
            is_prime[0:2] = [False, False]
            for num in range(2, int(math.isqrt(limit)) + 1):
                if is_prime[num]:
                    for multiple in range(num*num, limit + 1, num):
                        is_prime[multiple] = False
            primes = [num for num, prime in enumerate(is_prime) if prime]
            return primes
        
        # Generate primes up to sqrt(10^9) which is approximately 31623
        primes = sieve(31623)
        
        # Step 2: Function to compute square-free part of a number
        def get_square_free_part(num):
            d = 1
            for p in primes:
                if p*p > num:
                    break
                exponent = 0
                while num % p == 0:
                    num //= p
                    exponent += 1
                if exponent % 2 == 1:
                    d *= p
            if num > 1:
                d *= num
            return d
        
        # Step 3: Group numbers by their square-free parts and compute sums
        from collections import defaultdict
        group_sums = defaultdict(int)
        for num in nums:
            d = get_square_free_part(num)
            group_sums[d] += num
        
        # Step 4: Return the maximum sum among all groups
        return max(group_sums.values())