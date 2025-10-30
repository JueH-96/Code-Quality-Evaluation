import math
from typing import List

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        if not nums or not queries:
            return []
        
        M = max(nums)
        freq = [0] * (M + 1)
        for num in nums:
            freq[num] += 1
        
        F = [0] * (M + 1)
        for d in range(1, M + 1):
            total = 0
            k = d
            while k <= M:
                total += freq[k]
                k += d
            F[d] = total
        
        g = [0] * (M + 1)
        for d in range(M, 0, -1):
            if F[d] < 2:
                g[d] = 0
            else:
                g[d] = F[d] * (F[d] - 1) // 2
                j = 2 * d
                while j <= M:
                    g[d] -= g[j]
                    j += d
        
        H = [0] * (M + 1)
        for d in range(1, M + 1):
            H[d] = H[d - 1] + g[d]
        
        res = []
        for q in queries:
            lo, hi = 1, M
            ans_d = M
            while lo <= hi:
                mid = (lo + hi) // 2
                if H[mid] > q:
                    ans_d = mid
                    hi = mid - 1
                else:
                    lo = mid + 1
            res.append(ans_d)
        
        return res