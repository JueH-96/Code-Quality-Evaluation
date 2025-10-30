import math
from typing import List

MOD = 10**9 + 7

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        max_val = 200
        n_max = max_val
        mu = [1] * (n_max + 1)
        is_prime = [True] * (n_max + 1)
        primes = []
        is_prime[0] = is_prime[1] = False
        for i in range(2, n_max + 1):
            if is_prime[i]:
                primes.append(i)
                mu[i] = -1
            for p in primes:
                if i * p > n_max:
                    break
                is_prime[i * p] = False
                if i % p == 0:
                    mu[i * p] = 0
                    break
                else:
                    mu[i * p] = -mu[i]
        
        total_ans = 0
        n = len(nums)
        
        for h in range(1, max_val + 1):
            U = []
            for x in nums:
                if x % h == 0:
                    U.append(x // h)
            
            if len(U) == 0:
                continue
            
            M = max(U)
            freq = [0] * (M + 1)
            for x in U:
                if x <= M:
                    freq[x] += 1
            
            cnt = [0] * (M + 1)
            for d in range(1, M + 1):
                total_count = 0
                for k in range(d, M + 1, d):
                    total_count += freq[k]
                cnt[d] = total_count
            
            total_h = 0
            for d1 in range(1, M + 1):
                for d2 in range(1, M + 1):
                    g = math.gcd(d1, d2)
                    L = d1 * d2 // g
                    if L > M:
                        c1 = 0
                    else:
                        c1 = cnt[L]
                    c2 = cnt[d1] - c1
                    c3 = cnt[d2] - c1
                    nU = len(U)
                    c4 = nU - (c1 + c2 + c3)
                    
                    term1 = pow(3, c1, MOD) * pow(2, c2 + c3, MOD) % MOD
                    term2 = (pow(2, cnt[d1], MOD) + pow(2, cnt[d2], MOD) - 1) % MOD
                    term = (term1 - term2) % MOD
                    
                    total_h = (total_h + mu[d1] * mu[d2] * term) % MOD
            
            total_ans = (total_ans + total_h) % MOD
        
        return total_ans % MOD