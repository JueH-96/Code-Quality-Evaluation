from typing import List
from collections import defaultdict

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Smallest–prime–factor sieve up to n
        spf = list(range(n + 1))          # spf[x] = smallest prime factor of x
        for p in range(2, int(n ** 0.5) + 1):
            if spf[p] == p:               # p is prime
                for multiple in range(p * p, n + 1, p):
                    if spf[multiple] == multiple:
                        spf[multiple] = p
        
        # helper: square–free kernel of an integer (product of primes with odd exponent)
        def kernel(x: int) -> int:
            res = 1
            while x > 1:
                prime = spf[x]
                exp_parity = 0
                while x % prime == 0:
                    x //= prime
                    exp_parity ^= 1        # toggle parity
                if exp_parity:             # keep the prime if exponent is odd
                    res *= prime
            return res
        
        # group indices having identical kernels
        group_sum = defaultdict(int)       # kernel -> sum of nums at those indices
        for idx, val in enumerate(nums, start=1):   # 1-indexed enumeration
            k = kernel(idx)
            group_sum[k] += val
        
        # the answer is the largest group sum
        return max(group_sum.values())