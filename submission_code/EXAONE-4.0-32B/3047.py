import math
from collections import defaultdict

def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    i = 2
    while i * i <= n:
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
        i += 1
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    return primes

primes_list = sieve(35000)

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        groups = defaultdict(int)
        for num in nums:
            temp = num
            core = 1
            for p in primes_list:
                if p * p > temp:
                    break
                if temp % p == 0:
                    cnt = 0
                    while temp % p == 0:
                        cnt += 1
                        temp //= p
                    if cnt % 2 == 1:
                        core *= p
            if temp > 1:
                core *= temp
            groups[core] += num
        
        return max(groups.values()) if groups else 0