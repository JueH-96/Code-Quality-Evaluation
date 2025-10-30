from typing import List

class Solution:
    def minMaxSums(self, nums: List[int], k: int) -> int:
        mod = 10**9 + 7
        n = len(nums)
        
        inv = [0] * (k + 1)
        for j in range(1, k + 1):
            inv[j] = pow(j, mod - 2, mod)
        
        S = [0] * (n + 1)
        for m in range0, n + 1):
            end = min(k - 1, m)
            total = 1
            term = 1
            for j in range(1, end + 1):
                term = term * (m - j + 1) % mod
                term = term * inv[j] % mod
                total = (total + term) % mod
            S[m] = total
        
        nums.sort()
        total_min = 0
        total_max = 0
        for i in range(n):
            m_min = n - i - 1
            total_min = (total_min + nums[i] * S[m_min]) % mod
            
            m_max = i
            total_max = (total_max + nums[i] * S[m_max]) % mod
        
        return (total_min + total_max) % mod