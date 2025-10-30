from typing import List

class Solution:
    def minMaxSums(self, nums: List[int], k: int) -> int:
        mod = 10**9 + 7
        n = len(nums)
        nums.sort()
        
        # Precompute factorials and inverse factorials up to n
        fact = [1] * (n + 1)
        invfact = [1] * (n + 1)
        for i in range(1, n + 1):
            fact[i] = fact[i-1] * i % mod
        invfact[n] = pow(fact[n], mod-2, mod)
        for i in range(n, 0, -1):
            invfact[i-1] = invfact[i] * i % mod
        
        # Precompute for x=0..n-1: sums_s[x] = sum_{t=0..min(k-1, x)} C(x, t)
        # where C(x,t) = fact[x]/(fact[t]*fact[x-t])
        max_take = k-1
        sums_s = [0] * n
        for x in range(n):
            limit = min(max_take, x)
            s = 0
            for t in range(limit + 1):
                # C(x, t)
                s += fact[x] * invfact[t] % mod * invfact[x-t] % mod
            sums_s[x] = s % mod
        
        # Now sum contributions
        total = 0
        for i, v in enumerate(nums):
            # as max: choose up to k-1 from first i => sums_s[i]
            # as min: choose up to k-1 from last n-1-i => sums_s[n-1-i]
            contrib = (sums_s[i] + sums_s[n-1-i]) % mod
            total = (total + v * contrib) % mod
        
        return total