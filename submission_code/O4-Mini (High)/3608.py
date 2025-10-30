from typing import List
from math import gcd

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        n = len(nums)
        maxv = max(nums)
        
        # frequency of each value
        freq = [0] * (maxv + 1)
        for v in nums:
            freq[v] += 1
        
        # cnt[d] = number of elements in nums divisible by d
        cnt = [0] * (maxv + 1)
        for d in range(1, maxv + 1):
            for j in range(d, maxv + 1, d):
                cnt[d] += freq[j]
        
        # compute Möbius function up to maxv
        mu = [1] * (maxv + 1)
        mu[0] = 0
        is_prime = [True] * (maxv + 1)
        primes = []
        for i in range(2, maxv + 1):
            if is_prime[i]:
                primes.append(i)
                mu[i] = -1
            for p in primes:
                if i * p > maxv:
                    break
                is_prime[i * p] = False
                if i % p == 0:
                    mu[i * p] = 0
                    break
                else:
                    mu[i * p] = -mu[i]
        
        # precompute powers of 2 and 3
        max2 = 2 * n
        pow2 = [1] * (max2 + 1)
        for i in range(1, max2 + 1):
            pow2[i] = (pow2[i - 1] * 2) % mod
        pow3 = [1] * (n + 1)
        for i in range(1, n + 1):
            pow3[i] = (pow3[i - 1] * 3) % mod
        
        ans = 0
        # iterate over possible gcd g
        for g in range(1, maxv + 1):
            m_g = cnt[g]
            if m_g == 0:
                continue
            # maximum multiplier for g within range
            M = maxv // g
            # only d with mu[d] != 0 matter
            ds = [d for d in range(1, M + 1) if mu[d] != 0]
            # prefetch C_d = cnt[g*d]
            C_d = {d: cnt[g * d] for d in ds}
            
            total_g = 0
            # double sum over d,e with Möbius weights
            for d in ds:
                Cdi = C_d[d]
                if Cdi == 0:
                    continue
                mu_d = mu[d]
                for e in ds:
                    Cei = C_d[e]
                    if Cei == 0:
                        continue
                    mu_e = mu[e]
                    # lcm(d,e)
                    l = d * e // gcd(d, e)
                    Cl = cnt[g * l] if l <= M else 0
                    # number of assignments with d|gcd(A), e|gcd(B), both non-empty
                    expo1 = Cdi + Cei - 2 * Cl
                    t = (pow2[expo1] * pow3[Cl] - pow2[Cei] - pow2[Cdi] + 1) % mod
                    total_g = (total_g + mu_d * mu_e * t) % mod
            
            ans = (ans + total_g) % mod
        
        return ans