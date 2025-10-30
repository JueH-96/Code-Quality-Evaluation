from typing import List

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        def sieve_primes(n):
            is_prime = [True] * (n + 1)
            is_prime[0] = is_prime[1] = False
            for i in range(2, int(n**0.5) + 1):
                if is_prime[i]:
                    for j in range(i*i, n+1, i):
                        is_prime[j] = False
            primes = [i for i, val in enumerate(is_prime) if val]
            return primes
        
        # Precompute primes up to sqrt(1e9)
        max_prime = int(1e5**0.5) + 1  # Actually sqrt(1e9) is ~31622.776
        primes = sieve_primes(31622)
        
        from collections import defaultdict
        groups = defaultdict(int)
        
        for num in nums:
            if num == 0:
                s = 0  # Handle zero, though the constraints say nums[i] >=1
            else:
                temp = num
                s = 1
                for p in primes:
                    if p * p > temp:
                        break
                    if temp % p == 0:
                        exponent = 0
                        while temp % p == 0:
                            temp = temp // p
                            exponent += 1
                        if exponent % 2 == 1:
                            s *= p
                if temp > 1:
                    s *= temp
            groups[s] += num
        
        return max(groups.values())